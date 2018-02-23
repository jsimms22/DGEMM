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
/*//Works, but not fast
static void do_avx_unrolled (int lda, int M, int N, int K, double* a, double* b, double* c) {
  register __m128d cTmp, aTmp, bTmp;
  for (int j = 0; j < N; ++j) {
    for (int k = 0; k < K; ++k) {
      bTmp = _mm_load1_pd(b + k + j*lda);
      double* adda_mid = a + k*lda;
      double* addc_mid = c + j*lda;
      for (int i = 0; i < M/8*8; i += 8) {
        double* adda = adda_mid + i;
        double* addc = addc_mid + i;
        
	aTmp = _mm_loadu_pd(adda);
	cTmp = _mm_loadu_pd(addc);
	cTmp = _mm_add_pd(cTmp, _mm_mul_pd(bTmp, aTmp));
	_mm_storeu_pd(addc, cTmp);
	
	aTmp = _mm_loadu_pd(adda + 2);
	cTmp = _mm_loadu_pd(addc + 2);
	cTmp = _mm_add_pd(cTmp, _mm_mul_pd(bTmp, aTmp));
	_mm_storeu_pd((addc + 2), cTmp);

	aTmp = _mm_loadu_pd(adda + 4);
	cTmp = _mm_loadu_pd(addc + 4);
	cTmp = _mm_add_pd(cTmp, _mm_mul_pd(bTmp, aTmp));
	_mm_storeu_pd((addc + 4), cTmp);

	aTmp = _mm_loadu_pd(adda + 6);
	cTmp = _mm_loadu_pd(addc + 6);
	cTmp = _mm_add_pd(cTmp, _mm_mul_pd(bTmp, aTmp));
	_mm_storeu_pd((addc + 6), cTmp);
      }

      for (int i = M/8*8; i < M/2*2; i += 2) {
        double* adda = adda_mid + i;
        double* addc = addc_mid + i;
        
	aTmp = _mm_loadu_pd(adda);
	cTmp = _mm_loadu_pd(addc);
	cTmp = _mm_add_pd(cTmp, _mm_mul_pd(bTmp, aTmp));
	_mm_storeu_pd(addc, cTmp);
      }

      for (int i = M/2*2; i < M; ++i) {
        c[i + j*lda] += a[i + k*lda] * b[k + j*lda];
      }
    }
  }
}*/

static void do_avx256_unrolled (int lda, int M, int N, int K, double* a, double* b, double* c) {
  
}

static void do_avx256 (int lda, int M, int N, int K, double* a, double* b, double* c) {
    //printf("Did it declare?");
    __m256d m0,m1,m2,m3;
    //printf("Did it declare?");
    for (int i = 0; i < M; i += 4) {
      for (int j = 0; j < N; ++j) {
        m0 = _mm256_setzero_pd();  
        for (int k = 0; k < K; ++k) {
	  m1 = _mm256_load_pd(a+i+k*lda);
	  m2 = _mm256_broadcast_sd(b+k+j*lda);
	  m3 = _mm256_mul_pd(m1,m2);
	  m0 = _mm256_add_pd(m0,m3);
        }
        m1 = _mm256_load_pd(c+i+j*lda);
        m0 = _mm256_add_pd(m0,m1);
        _mm256_storeu_pd(c+i+j*lda,m0);
      }
    }
}

static inline void do_simple (int lda, int M, int N, int K, double* a, double* b, double* c) {
    //printf("Did it do else?");
    // For each row of A
    for (int i = 0; i < M; ++i) {
      // For each column of B
      for (int j = 0; j < N; ++j) {
        // Compute C[i,j] 
        register double cij = 0.0;
        for (int k = 0; k < K; ++k){
          cij += a[i+k*lda] * b[k+j*lda];
	}
        c[i+j*lda] += cij;
      }
    }
}
/* This auxiliary subroutine performs a smaller dgemm operation
 *  C := C + A * B
 * where C is M-by-N, A is M-by-K, and B is K-by-N. */
static void do_block (int lda, int M, int N, int K, double* A, double* B, double* C)
{
  //printf("M = %d\t,N = %d,K = %d, \n",M,N,K);
  //printf("Did it declare?");
  register char status;
  if (M == N) {
    if (K == N) {
      if (M == K) { 
        if (M%4 == 0) {
	  if (N%4 == 0) {
	    if (K%4 == 0) {
	      status = '0';
	    }
	  }
	} else if (M%2 == 0) {
	    if (N%2 == 0) {
	      if (K%2 == 0) {
		status = '1';
	      }
	    }
	  } else {
	  status = '2';
	}
      }
    }
  }
  //printf("Did it declare?");
  if (status == '0') {
    do_avx256 (lda, M, N, K, A, B, C);
  } else if (status == '1') {
    do_simple (lda, M, N, K, A, B, C);
  } else {
    do_simple (lda, M, N, K, A, B, C);
  }
  //printf("exiting a block\n");
 
  //Does not run fast
  //do_avx_unrolled (lda, M, N, K, A, B, C);
}
/* This routine performs a dgemm operation
 *  C := C + A * B
 * where A, B, and C are lda-by-lda matrices stored in column-major format. 
 * On exit, A and B maintain their input values. */
void square_dgemm (/*int iii,*/int lda, double* A, double* B, double* C)
{
  //printf("Do you see me now at %d\t \n",lda);
  //Block size in 1D for L2 cache - BLOCK2 - for L1 cache - BLOCK 1 - 
  int BLOCK1 = 32;
  int BLOCK2 = 64;
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
	      //printf("M = %d\t, N = %d, K = %d \n",M,N,K);
	      //printf("i = %d\t, j = %d, k = %d \n",i,j,k); 
              do_block(lda, M, N, K, A + i + k*lda, B + k + j*lda, C + i + j*lda);
            }
          }
        }
      }
    }
  }
}
