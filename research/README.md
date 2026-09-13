# research/ — 外部调研留档

## 规则

**只入库人写的结论**(带证据 URL),**不入库原始 API payload** —— 它们可重新拉取,
而且单个 tree dump 就有 4 MB。项目规则是「git 只有代码/文档/probes」。

被 gitignore 的:`*.json`、`d*/`(各 PR 的 diff 解包)。
要重取,按 `PLE_INTEGRATION_SURFACE.md` 里每条的 evidence URL 走 GitHub REST API:

```bash
curl -s https://api.github.com/repos/vllm-project/vllm/pulls/54129          # PR 正文
curl -s https://api.github.com/repos/vllm-project/vllm/pulls/54070/files    # 文件清单
curl -s "https://api.github.com/repos/sgl-project/sglang/git/trees/main?recursive=1"
```

> **注意**:GitHub 的 HTML 页面取回来几乎全是导航框架(实测被截断)。
> **用 `api.github.com`,不要用 `github.com`**;文件内容用
> `raw.githubusercontent.com/<org>/<repo>/<ref>/<path>`。

## 内容

| 文件 | 内容 |
|---|---|
| `PLE_INTEGRATION_SURFACE.md` | vLLM + SGLang 的 PLE / n-gram gather 集成面:gather 归属、graph 位置、offload 各设计是否需断点、最小改动点。每条标 [C] 已读 / [U] 未确认 |

## 一条方法论

本目录存在的原因是 Session 44 的一次**纪律失败**:此前两次断言「SGLang 没有 Engram/PLE」——
对 0.5.19 为真,**对 `main` 为假**。根因是**从已安装的 release + 一条关于 branch 的 bug 报告
去推理上游主线**,而不是读 `main`。

⇒ 已写入 §36.6 **第九条纪律**:关于上游能力的断言,必须落到 **main / 目标 branch 的具体 path**,
并注明 commit 或 tree 来源。
