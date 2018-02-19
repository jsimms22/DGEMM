/* 
 *     Please include compiler name below (you may also include any other modules you would like to be loaded)
 *
 *     COMPILER= gnu
 *
 *         Please include All compiler flags and libraries as you want them run. You can simply copy this over from the Makefile's first few lines
 *          
 *          CC = cc
 *          OPT = -O3
 *          CFLAGS = -Wall -std=gnu99 $(OPT)
 *          MKLROOT = /opt/intel/composer_xe_2013.1.117/mkl
 *          LDLIBS = -lrt -Wl,--start-group $(MKLROOT)/lib/intel64/libmkl_intel_lp64.a $(MKLROOT)/lib/intel64/libmkl_sequential.a $(MKLROOT)/lib/intel64/libmkl_core.a -Wl,--end-group -lpthread -lm
 *
 *          */

#include <stdlib.h>
#include <time.h>
#include <immintrin.h>
#include <pmmintrin.h>
#include <xmmintrin.h>
#include <emmintrin.h>
const char* dgemm_desc = "Simple blocked dgemm.";

#define min(a,b) (((a)<(b))?(a):(b))
#define likely(x)       __builtin_expect((x),1)
#define unlikely(x)     __builtin_expect((x),0)

#define ARRAY(A,i,j) (A)[(j)*lda + (i)]
/* This auxiliary subroutine performs a smaller dgemm operation
 *  C := C + A * B
 * where C is M-by-N, A is M-by-K, and B is K-by-N. */
static inline void do_block (int lda, int M, int N, int K, double* A, double* B, double* C)
{
  printf("K = %d\t,M = %d,N = %d, \n",K,M,N);
  //Do math here
  int fringe1 = M % 4;
  int fringe2 = N % 4;
  int fringe3 = K % 4;
  if (fringe1 == 0 && fringe2 == 0 && fringe3 == 0 && K == M && M == N && N == K) {
    __m256d m0,m1,m2,m3;
    for (int i = 0; i < M; i += 4) {
      for (int j = 0; j < N; ++j) {
        m0 = _mm256_setzero_pd();  
        for (int k = 0; k < K; ++k) {
	  m1 = _mm256_load_pd(A+i+k*lda);
	  m2 = _mm256_broadcast_sd(B+k+j*lda);
	  m3 = _mm256_mul_pd(m1,m2);
	  m0 = _mm256_add_pd(m0,m3);
        }
        _mm256_store_pd(C+i+j*lda,m0);
      }
    }
  } else {
    // For each row of A
    for (int i = 0 ; i < M; ++i) {
      // For each column of B
      for (int j = 0; j < N; ++j) {
        // Compute C[i,j] 
        double cij = 0.0;
        for (int k = 0; k < K; ++k){
          cij += A[i+k*lda] * B[k+j*lda];
	}
        C[i+j*lda] = cij;
      }
    }
  }
  printf("exiting a block\n"); 
}

/* This routine performs a dgemm operation
 *  C := C + A * B
 * where A, B, and C are lda-by-lda matrices stored in column-major format. 
 * On exit, A and B maintain their input values. */
void square_dgemm (/*int iii,*/int lda, double* A, double* B, double* C)
{
  printf("Do you see me now at %d\t \n",lda);
  //This is to print out a small n*n C matrix (before calculation)
  /*if (lda == 8) {
    for (int i = 0; i < lda; ++i) {
      for (int k = 0; k < lda; ++k) {
        printf("%g\t ",C[i+lda*k]);
      }
      printf("\n");
    }
  }*/

  //Block size in 1D for L2 cache - BLOCK2 - for L1 cache - BLOCK 1 - 
  int BLOCK1 = 64;
  int BLOCK2 = 128;
  
    
  //Proposed blocking method for 2 levels of memory - L1 and L2
  for (int x = 0; x < lda; x += BLOCK2) {
    int lim_k = x + min (BLOCK2,lda-x);
    for (int y = 0; y < lda; y += BLOCK2) {
      int lim_j = y + min (BLOCK2,lda-y);
      for (int z = 0; z < lda; z += BLOCK2) {
	int lim_i = z + min (BLOCK2,lda-z);
        for (int k = x; k < lim_k; k += BLOCK1) {
	  int K = min (BLOCK1,lim_k-k);
	  for (int j = y; j < lim_j; j += BLOCK1) {
	    int N = min (BLOCK1,lim_j-j);
	    for (int i = z; i < lim_i; i += BLOCK1) {
	      int M = min (BLOCK1,lim_i-i);
	      printf("M = %d\t, N = %d, K = %d \n",M,N,K);
	      //printf("i = %d\t, j = %d, k = %d \n",i,j,k); 
              do_block(lda, M, N, K, A + i + k*lda, B + k + j*lda, C + i + j*lda);
            }
          }
        }
      }
    }
  }
 

  //This is to print out a small n*n C matrix (after calculation)
  /*if (lda == 96) {
    for (int i = 0; i < lda; ++i) {
      for (int k = 0; k < lda; ++k) { 
	printf("%g\t ",C[i+lda*k]);
      }
      printf("\n");
    }
  }*/
  printf("\n");
}
