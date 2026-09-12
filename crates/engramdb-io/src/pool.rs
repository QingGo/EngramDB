//! 常驻工作线程池：把 `gather_pp` / `read_records_parallel` 的**每次调用 spawn**
//! 换成一次投递 + 唤醒。
//!
//! # 为什么必须做（实测依据）
//!
//! `docs/roadmap.md` §33.1 在本机（AutoDL 容器，128 线程，宿主 load ≈11）量到
//! **单次 `std::thread::spawn` 约 30–35 μs，且从 n=4 起严格线性**：
//!
//! | spawn 数 | 0 | 1 | 4 | 8 | 16 | 32 | 64 |
//! |---|---|---|---|---|---|---|---|
//! | μs/次调用 | 0.2 | 61.5 | 136.0 | 225.6 | 459.2 | **1025.1** | 2221.6 |
//!
//! 而 `gather_pp` 在 65 shard、`threads=32` 时每次调用 spawn 约 22 个线程
//! ⇒ **≈0.7 ms/次调用**。这与小 batch 的端到端实测吻合
//! （V4.1 几何 tokens=1：t=1 时 223 μs/call，t=32 时 515 μs/call）。
//!
//! # 安全性
//!
//! [`scope_run`] 接受带非 `'static` 生命周期的任务，靠**作用域退出前必定跑完所有任务**
//! 来保证借用有效。实现上把 `Box<dyn FnOnce() + Send + 'a>` 的生命周期参数擦除成
//! `'static` 后投递；`scope_run` 只在完成计数达到任务数后才返回，且任务 Box 在
//! 计数递增**之前**就已执行并释放，因此不存在悬垂。这是 rayon / crossbeam 的同一套做法。
//!
//! 任务 panic 会被 worker 捕获、计数照常递增、并在 [`scope_run`] 里于调用线程重放 ——
//! 语义与 `std::thread::scope` 一致（panic 本身不会让池子丢线程或死锁）。

use std::any::Any;
use std::collections::VecDeque;
use std::panic::{catch_unwind, resume_unwind, AssertUnwindSafe};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex, OnceLock};

type Job = Box<dyn FnOnce() + Send + 'static>;

/// 池子上限。实测 32 线程已是收益拐点（`roadmap.md` §32.2），留一倍余量即可，
/// 避免在 128 核机器上为一个库起 128 个常驻线程。
const MAX_WORKERS: usize = 64;

struct Inner {
    queue: Mutex<VecDeque<Job>>,
    cv: Condvar,
    shutdown: AtomicBool,
}

struct Pool {
    inner: Arc<Inner>,
    workers: Vec<std::thread::JoinHandle<()>>,
}

impl Pool {
    fn new(n: usize) -> std::io::Result<Self> {
        let n = n.max(1);
        let inner = Arc::new(Inner {
            queue: Mutex::new(VecDeque::new()),
            cv: Condvar::new(),
            shutdown: AtomicBool::new(false),
        });
        let mut workers = Vec::with_capacity(n);
        for i in 0..n {
            let inner = Arc::clone(&inner);
            let h = std::thread::Builder::new()
                .name(format!("engramdb-io-{i}"))
                .spawn(move || worker_loop(&inner))?;
            workers.push(h);
        }
        Ok(Self { inner, workers })
    }

    /// 一次加锁投递整批（比逐个 `submit` 少 n−1 次锁/唤醒往返）。
    fn submit_many(&self, jobs: Vec<Job>) {
        if jobs.is_empty() {
            return;
        }
        let n = jobs.len();
        let mut q = self.inner.queue.lock().unwrap();
        q.extend(jobs);
        drop(q);
        // **精确唤醒 n 个**，不要 `notify_all`。
        //
        // 这里踩过一次坑（roadmap §33.2）：`notify_all` 会把全部 64 个 worker 唤醒，
        // 而每个任务完成时又会再唤醒一轮 —— 22 个任务 × 64 = 上千次伪唤醒，
        // 做 IO 的线程被挤下 CPU，**冷读边际成本从 3.9 涨到 7.8 μs/行**。
        // 开销看着没变，介质却"变慢"了一倍。
        for _ in 0..n {
            self.inner.cv.notify_one();
        }
    }
}

impl Drop for Pool {
    fn drop(&mut self) {
        self.inner.shutdown.store(true, Ordering::SeqCst);
        self.inner.cv.notify_all();
        for w in self.workers.drain(..) {
            let _ = w.join();
        }
    }
}

fn worker_loop(inner: &Arc<Inner>) {
    loop {
        let job = {
            let mut q = inner.queue.lock().unwrap();
            loop {
                if let Some(j) = q.pop_front() {
                    break Some(j);
                }
                if inner.shutdown.load(Ordering::SeqCst) {
                    break None;
                }
                q = inner.cv.wait(q).unwrap();
            }
        };
        match job {
            // 任务 panic 不能杀掉 worker（否则池子会逐渐缩水至死锁），
            // 记在任务自己的包装里由调用方重放。
            Some(j) => {
                let _ = catch_unwind(AssertUnwindSafe(j));
            }
            None => return,
        }
    }
}

/// 全局池：进程内惰性创建、只创建一次。
///
/// 创建失败（例如线程数受限）时返回 `None`，调用方回退到 `std::thread::scope`。
fn global() -> Option<&'static Pool> {
    static POOL: OnceLock<Option<Pool>> = OnceLock::new();
    POOL.get_or_init(|| {
        // ⚠️ 这里踩过一个很贵的坑（roadmap §33.2）：
        // `available_parallelism()` **遵守 cgroup CPU 配额**，在本机返回的是 **16**，
        // 而不是 `nproc` 的 128。第一版直接用它当池子大小 ⇒ 池子只有 16 个 worker，
        // 而 t=32 每次调用要提交 22 个任务 ⇒ 分两波跑 ⇒ **IO 并发度腰斩，冷读慢 1.7×**
        // （边际 4.5 → 8.0 μs/行）。这与线程放置/惊群都无关，纯粹是池子太小。
        //
        // 对**IO 密集**负载，并发度不该被 CPU 配额限制：实测 t=32 仍显著优于 t=16
        // （§32.2）。所以下界取 `DEFAULT_GATHER_THREADS`。
        let auto = std::thread::available_parallelism()
            .map(|x| x.get())
            .unwrap_or(4)
            .max(crate::batch::DEFAULT_GATHER_THREADS)
            .clamp(1, MAX_WORKERS);
        // `ENGRAMDB_POOL_WORKERS` 仅用于 A/B 实验。
        let n = std::env::var("ENGRAMDB_POOL_WORKERS")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .filter(|&v| v > 0)
            .unwrap_or(auto);
        Pool::new(n).ok()
    })
    .as_ref()
}

/// 池子里实际有多少 worker（诊断用；池不可用时为 0）。
pub fn workers() -> usize {
    global().map(|p| p.workers.len()).unwrap_or(0)
}

/// 池子是否被环境变量关掉。
///
/// `ENGRAMDB_NO_POOL=1` 时 [`scope_run`] 退回 `std::thread::scope`（即池子之前的行为）。
/// 存在的理由有两个：一是**同机同刻 A/B**（否则无法把差异归因到池子而不是机器噪声），
/// 二是万一池子在某环境上有问题，有不改代码的退路。
pub fn disabled() -> bool {
    static D: OnceLock<bool> = OnceLock::new();
    *D.get_or_init(|| {
        std::env::var_os("ENGRAMDB_NO_POOL").is_some_and(|v| v != "0" && !v.is_empty())
    })
}

/// 在常驻池上跑完 `jobs` 再返回。
///
/// 语义等同 `std::thread::scope(|s| for j in jobs { s.spawn(j) })`，
/// 但**不创建线程**。若全局池不可用（或 `ENGRAMDB_NO_POOL=1`），自动回退到
/// `std::thread::scope`，因此调用方无需处理失败。
///
/// # Panic
///
/// 任一任务 panic：所有任务仍会跑完，随后在调用线程上重放**第一个** panic。
pub fn scope_run<'a>(jobs: Vec<Box<dyn FnOnce() + Send + 'a>>) {
    if jobs.is_empty() {
        return;
    }
    let pool = if disabled() { None } else { global() };
    let Some(pool) = pool else {
        // 回退：与旧行为完全一致
        std::thread::scope(|s| {
            for j in jobs {
                s.spawn(j);
            }
        });
        return;
    };
    // 任务数超过 worker 数时**主动回退**到 `thread::scope`：
    // 池子跑不动这么多并发，分波执行会把 IO 并发度砍掉（§33.2 的教训）。
    // 宁可付 spawn 的钱，也不要悄悄降并发。
    if jobs.len() > pool.workers.len() {
        std::thread::scope(|s| {
            for j in jobs {
                s.spawn(j);
            }
        });
        return;
    }

    let n = jobs.len();
    let done = Arc::new((Mutex::new(0usize), Condvar::new()));
    let panics: Arc<Mutex<Vec<Box<dyn Any + Send>>>> = Arc::new(Mutex::new(Vec::new()));

    let mut wrapped: Vec<Job> = Vec::with_capacity(n);
    for job in jobs {
        let done = Arc::clone(&done);
        let panics = Arc::clone(&panics);
        // SAFETY: 生命周期擦除。`scope_run` 在下面等到 `*count == n` 才返回，
        // 而每个包装任务在**递增计数之前**就已经执行完 `job` 并释放其 Box，
        // 所以 `'a` 借用的数据一定活过任务执行期。任务不会逃逸到池子的队列之外。
        let job: Job = unsafe {
            std::mem::transmute::<Box<dyn FnOnce() + Send + 'a>, Box<dyn FnOnce() + Send + 'static>>(
                job,
            )
        };
        wrapped.push(Box::new(move || {
            let r = catch_unwind(AssertUnwindSafe(job));
            {
                let (m, cv) = &*done;
                let mut g = m.lock().unwrap();
                *g += 1;
                cv.notify_all();
            }
            if let Err(e) = r {
                panics.lock().unwrap().push(e);
            }
        }));
    }
    pool.submit_many(wrapped);

    // 等全部完成（含 panic 的任务也会递增计数）
    {
        let (m, cv) = &*done;
        let mut g = m.lock().unwrap();
        while *g < n {
            g = cv.wait(g).unwrap();
        }
    }

    let first = panics.lock().unwrap().drain(..).next();
    if let Some(p) = first {
        resume_unwind(p);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicUsize;

    #[test]
    fn runs_every_job_exactly_once() {
        let c = Arc::new(AtomicUsize::new(0));
        let jobs: Vec<Box<dyn FnOnce() + Send>> = (0..64)
            .map(|_| {
                let c = Arc::clone(&c);
                Box::new(move || {
                    c.fetch_add(1, Ordering::SeqCst);
                }) as Box<dyn FnOnce() + Send>
            })
            .collect();
        scope_run(jobs);
        assert_eq!(c.load(Ordering::SeqCst), 64);
    }

    /// 关键用例：任务借用栈上的数据（非 `'static`），scope_run 返回后借用必须仍然有效。
    #[test]
    fn borrows_non_static_data() {
        let src = vec![1u8, 2, 3, 4, 5, 6, 7, 8];
        let mut dst = vec![0u8; 8];
        let src_ref = &src;
        let dst_ref = &mut dst;
        let jobs: Vec<Box<dyn FnOnce() + Send + '_>> = dst_ref
            .chunks_mut(2)
            .enumerate()
            .map(|(i, chunk)| {
                Box::new(move || {
                    chunk.copy_from_slice(&src_ref[i * 2..i * 2 + 2]);
                }) as Box<dyn FnOnce() + Send + '_>
            })
            .collect();
        scope_run(jobs);
        assert_eq!(dst, src);
    }

    #[test]
    fn empty_is_noop() {
        let jobs: Vec<Box<dyn FnOnce() + Send>> = Vec::new();
        scope_run(jobs);
    }

    #[test]
    fn panic_propagates_and_pool_survives() {
        let c = Arc::new(AtomicUsize::new(0));
        let jobs: Vec<Box<dyn FnOnce() + Send>> = (0..8)
            .map(|i| {
                let c = Arc::clone(&c);
                Box::new(move || {
                    c.fetch_add(1, Ordering::SeqCst);
                    if i == 3 {
                        panic!("boom");
                    }
                }) as Box<dyn FnOnce() + Send>
            })
            .collect();
        let r = catch_unwind(AssertUnwindSafe(|| scope_run(jobs)));
        assert!(r.is_err(), "panic 必须传播到调用线程");
        // 所有任务仍然跑完（含 panic 的那个）
        assert_eq!(c.load(Ordering::SeqCst), 8);
        // 池子没死：还能继续用
        let c2 = Arc::new(AtomicUsize::new(0));
        let c3 = Arc::clone(&c2);
        scope_run(vec![Box::new(move || {
            c3.fetch_add(1, Ordering::SeqCst);
        }) as Box<dyn FnOnce() + Send>]);
        assert_eq!(c2.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn repeated_use_is_stable() {
        for _ in 0..200 {
            let c = Arc::new(AtomicUsize::new(0));
            let c2 = Arc::clone(&c);
            scope_run(vec![Box::new(move || {
                c2.fetch_add(1, Ordering::SeqCst);
            }) as Box<dyn FnOnce() + Send>]);
            assert_eq!(c.load(Ordering::SeqCst), 1);
        }
    }
}
