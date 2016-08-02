/* 
 * compile: 
 gcc -Wall -O3 -mavx a.c -fopenmp -funroll-loops
 * run:
 OMP_NUM_THREADS=XX numactl --physcpubind=.... ./a.out array_size_in_bytes number_of_times_each_thread_scan_it
 */

#include <assert.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/mman.h>

typedef unsigned long long ull;
static inline ull rdtsc() {
  ull u;
  asm volatile ("rdtsc;shlq $32,%%rdx;orq %%rdx,%%rax":"=a"(u)::"%rdx");
  return u;
}

double cur_time() {
  struct timespec tp[1];
  int r = clock_gettime(CLOCK_REALTIME, tp);
  assert(r == 0);
  return tp->tv_sec + tp->tv_nsec * 1.0e-9;
}

typedef float float8 __attribute__((vector_size(32)));

void scan(long nbytes, long niters, double * dtp, ull * dcp) {
  float * c;
#if 1
  int r = posix_memalign((void **)&c, 4096, nbytes);
  assert(r == 0);
#else
  c = mmap(0, nbytes, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS|MAP_POPULATE, -1, 0);
  if (c == MAP_FAILED) {
    perror("mmap");
    exit(1);
  }
#endif
  long i;
  /* initialize the float array randomly */
  unsigned short rg[3] = { 1, 2, 3 };
  long n_floats = nbytes / sizeof(float);
  for (i = 0; i < n_floats; i++) {
    c[i] = erand48(rg);
  }

  /* access the array, 32 bytes at a time using avx */
  long n = nbytes / sizeof(float8);
  float8 x = { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 };
  volatile float8 * a = (volatile float8 *)c;
  long t;
#pragma omp barrier
  double t0 = cur_time();
  ull c0 = rdtsc();
  for (t = 0; t < niters; t++) {
    long i;
    asm volatile("# begin");
    for (i = 0; i < n; i++) {
      a[i] += x;
    }
    asm volatile("# end");
  }
#pragma omp barrier
  ull c1 = rdtsc();
  double t1 = cur_time();
  *dtp = t1 - t0;
  *dcp = c1 - c0;
  //printf("x = %f\n", x[0]);
}

int main(int argc, char ** argv) {
  long nbytes = atol(argv[1]);
  long niters = atol(argv[2]);
  int nth = omp_get_max_threads();
  printf("%d threads each scanning %ld bytes %ld times\n", 
	 nth, nbytes, niters);
  double dt;
  ull dc;
#pragma omp parallel
  {
    double dt_;
    ull dc_;
    scan(nbytes, niters, &dt_, &dc_);
#pragma omp master
    {
      dt = dt_;
      dc = dc_;
    }
  }
  double bytes_total = (double)nbytes * (double)niters * (double)nth;
  printf("%.0f bytes scanned in total in %.3f sec / %llu clocks\n",
	 
	 bytes_total, dt, dc);
  printf("%.3f GB/sec\n", bytes_total * 1.0e-9 / dt);
  printf("%.1f clocks/cache line\n",  ((double)dc * 64.0) / bytes_total);
  return 0;
}
