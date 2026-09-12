#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <pthread.h>
#include <time.h>
#include <stdint.h>
#include <errno.h>

#define BLK 4096

static double now_s(void){struct timespec ts;clock_gettime(CLOCK_MONOTONIC,&ts);return ts.tv_sec+ts.tv_nsec*1e-9;}

static uint64_t S0,S1;
static inline uint64_t rnd(void){uint64_t x=S0,y=S1;S0=y;x^=x<<23;S1=x^y^(x>>17)^(y>>26);return S1+y;}
static void rseed(uint64_t s){S0=s^0x9E3779B97F4A7C15ULL;S1=s+0xBF58476D1CE4E5B9ULL;for(int i=0;i<32;i++)rnd();}

typedef struct{int rc;double us;}res_t;
typedef struct{int fd;uint64_t*offs;long n;res_t*res;long base;}task_t;

static void*worker(void*a){
  task_t*t=a; void*buf;
  if(posix_memalign(&buf,BLK,BLK)){fprintf(stderr,"memalign fail\n");return NULL;}
  for(long i=0;i<t->n;i++){
    double t0=now_s();
    ssize_t r=pread(t->fd,buf,BLK,(off_t)t->offs[i]);
    double t1=now_s();
    if(r!=BLK){t->res[t->base+i].rc=(int)r;t->res[t->base+i].us=-1.0;
      fprintf(stderr,"pread r=%zd errno=%d off=%llu\n",r,errno,(unsigned long long)t->offs[i]);break;}
    t->res[t->base+i].rc=BLK; t->res[t->base+i].us=(t1-t0)*1e6;
  }
  free(buf); return NULL;
}
static int cmp(const void*a,const void*b){double x=((const res_t*)a)->us,y=((const res_t*)b)->us;return x<y?-1:(x>y?1:0);}

static void run(const char*path,uint64_t filesz,long nreads,int conc,int direct,uint64_t seed,const char*tag){
  int fd=open(path,O_RDONLY|(direct?O_DIRECT:0));
  if(fd<0){fprintf(stderr,"open %s: %s\n",path,strerror(errno));return;}
  uint64_t nblk=filesz/BLK;
  uint64_t*offs=malloc(sizeof(uint64_t)*nreads);
  unsigned char*seen=calloc(nblk,1);
  if(!offs||!seen){fprintf(stderr,"alloc fail\n");return;}
  rseed(seed);
  long got=0; long tries=0;
  while(got<nreads && tries<(long)nreads*30){
    uint64_t b=rnd()%nblk; tries++;
    if(seen[b])continue; seen[b]=1; offs[got++]=b*BLK;
  }
  free(seen);
  nreads=got;
  res_t*res=calloc(nreads,sizeof(res_t));
  pthread_t*th=malloc(sizeof(pthread_t)*conc);
  task_t*tk=malloc(sizeof(task_t)*conc);
  long per=(nreads+conc-1)/conc;
  double t0=now_s();
  for(int i=0;i<conc;i++){
    long b=i*per,n=per; if(b>=nreads)n=0; else if(b+n>nreads)n=nreads-b;
    tk[i]=(task_t){fd,offs,n,res,b}; pthread_create(&th[i],NULL,worker,&tk[i]);
  }
  for(int i=0;i<conc;i++)pthread_join(th[i],NULL);
  double tot=(now_s()-t0)*1e6;
  long ok=0; double sum=0;
  for(long i=0;i<nreads;i++) if(res[i].us>=0.0){ok++;sum+=res[i].us;}
  if(ok==0){printf("%-20s c=%-3d FAILED\n",tag,conc);goto done;}
  qsort(res,nreads,sizeof(res_t),cmp);
  { long i50=ok/2, i99=(long)(ok*0.99); if(i99>=ok)i99=ok-1;
    printf("%-20s c=%-3d %-8s n=%-6ld wall=%8.1fms mean=%8.2f us/read p50=%8.2f p99=%9.2f thr=%7.1f MB/s\n",
      tag,conc,direct?"O_DIRECT":"buffered",ok,tot/1000.0,sum/ok,res[i50].us,res[i99].us,
      (double)ok*BLK/1048576.0/(tot/1e6)); }
done:
  free(offs);free(res);free(th);free(tk); close(fd); fflush(stdout);
}

int main(int argc,char**argv){
  const char*p=argv[1]; uint64_t sz=strtoull(argv[2],NULL,10); long n=atol(argv[3]);
  printf("== probe %s  size=%.1f GiB  nreads/config=%ld  BLK=%d ==\n",p,(double)sz/1073741824.0,n,BLK);
  printf("-- O_DIRECT (bypasses page cache by construction => TRUE device latency) --\n"); fflush(stdout);
  run(p,sz,n,1,1,1001,"direct-c1");
  run(p,sz,n,8,1,2001,"direct-c8");
  run(p,sz,n,32,1,3001,"direct-c32");
  printf("-- buffered, FRESH disjoint offsets, each read exactly once (EngramDB's real path) --\n"); fflush(stdout);
  run(p,sz,n,1,0,4001,"buffered-cold-c1");
  run(p,sz,n,8,0,5001,"buffered-cold-c8");
  run(p,sz,n,32,0,6001,"buffered-cold-c32");
  printf("-- MECHANICAL SELF-CHECK: same seed twice, back to back --\n");
  printf("   (if 'warm' is not much faster than 'cold', the cold methodology is BROKEN)\n"); fflush(stdout);
  run(p,sz,n,1,0,7777,"SELFCHECK-cold");
  run(p,sz,n,1,0,7777,"SELFCHECK-warm");
  return 0;
}
