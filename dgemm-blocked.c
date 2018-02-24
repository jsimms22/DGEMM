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
 /*C Matrix 8x8			A Matrix   B Matrix
 * | 00 10 20 30 40 50 60 70 |  | 0x -> |  | 0x 1x 2x 3x 4x 5x 6x 7x |
 * | 01 11 21 31 41 51 61 71 |  | 1x -> |  |                         |
 * | 02 12 22 32 42 52 62 72 |  | 2x -> |  |                         |
 * | 03 13 23 33 43 53 63 73 |  | 3x -> |  |                         |
 * | 04 14 24 34 44 54 64 74 |  | 4x -> |  |                         |
 * | 05 15 25 35 45 55 65 75 |  | 5x -> |  |                         |
 * | 06 16 26 36 46 56 66 76 |  | 6x -> |  |                         |
 * | 07 17 27 37 47 57 67 77 |  | 7x -> |  |                         |
 */

static void do_avx256_unrolled (int lda, int K, double* a, double* b, double* c) {
  //printf("Can you see me");
  __m256d a0x_3x, /*a4x_7x,*/
    bx0, bx1, bx2, bx3,/* bx4, bx5, bx6, bx7*/
    c00_30, /*c40_70,*/
    c01_31, /*c41_71,*/
    c02_32, /*c42_72,*/
    c03_33/*, c43_73*/;
  
  double* c01_31_ptr = c + lda;
  double* c02_32_ptr = c01_31_ptr + lda;
  double* c03_33_ptr = c02_32_ptr + lda;
  
  c00_30 = _mm256_loadu_pd(c);
  //c40_70 = _mm256_loadu_pd(c + 4);
  c01_31 = _mm256_loadu_pd(c01_31_ptr);
  //c41_71 = _mm256_loadu_pd(c + lda + 4);
  c02_32 = _mm256_loadu_pd(c02_32_ptr);
  //c42_72 = _mm256_loadu_pd(c + 2*lda + 4);
  c03_33 = _mm256_loadu_pd(c03_33_ptr);
  //c43_73 = _mm256_loadu_pd(c + 3*lda + 4);
  
  for (int x = 0; x < K; ++x) {
    a0x_3x = _mm256_load_pd(a);
    //a4x_7x = _mm256_load_pd(a+4);
    a += 4;

    bx0 = _mm256_broadcast_sd(b++);
    bx1 = _mm256_broadcast_sd(b++);
    bx2 = _mm256_broadcast_sd(b++);
    bx3 = _mm256_broadcast_sd(b++);
    //bx4 = _mm256_broadcast_sd(b++);
    //bx5 = _mm256_broadcast_sd(b++);
    //bx6 = _mm256_broadcast_sd(b++);
    //bx7 = _mm256_broadcast_sd(b++);

    c00_30 = _mm256_add_pd(c00_30, _mm256_mul_pd(a0x_3x,bx0));
    c01_31 = _mm256_add_pd(c01_31, _mm256_mul_pd(a0x_3x,bx1));
    c02_32 = _mm256_add_pd(c02_32, _mm256_mul_pd(a0x_3x,bx2));
    c03_33 = _mm256_add_pd(c03_33, _mm256_mul_pd(a0x_3x,bx3));
  }
  
  _mm256_storeu_pd(c,c00_30);
  _mm256_storeu_pd(c01_31_ptr,c01_31);
  _mm256_storeu_pd(c02_32_ptr,c02_32);
  _mm256_storeu_pd(c03_33_ptr,c03_33);
}

static inline void copy_a (int lda, const int K, double* a_src, double* a_dest) {
  for (int i = 0; i < K; ++i) {
    *a_dest++ = *a_src;
    *a_dest++ = *(a_src + 1);
    *a_dest++ = *(a_src + 2);
    *a_dest++ = *(a_src + 3);
    a_src += lda;
  }
}

static inline void copy_b (int lda, const int K, double* b_src, double* b_dest) {
  double *b_ptr0, *b_ptr1, *b_ptr2, *b_ptr3;
  b_ptr0 = b_src;
  b_ptr1 = b_ptr0 + lda;
  b_ptr2 = b_ptr1 + lda;
  b_ptr3 = b_ptr2 + lda;

  for (int i = 0; i < K; ++i) {
    *b_dest++ = *b_ptr0++;
    *b_dest++ = *b_ptr1++;
    *b_dest++ = *b_ptr2++;
    *b_dest++ = *b_ptr3++;
  }
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
static inline void do_block (int lda, int M, int N, int K, double* A, double* B, double* C)
{
  //printf("Did it enter do_block?\n");
  double A_block[M*K], B_block[K*N];
  double *a_ptr, *b_ptr, *c;

  const int Nmax = N-3;
  int Mmax = M-3;
  int fringe1 = M%4;
  int fringe2 = N%4;
  
  int i = 0, j = 0, p = 0;
    
  for (j = 0; j < Nmax; j += 4) {
    b_ptr = &B_block[j*K];
    copy_b (lda, K, B + j*lda, b_ptr);
    for (i = 0; i < Mmax; i += 4) {
      //printf("j = %d\t, i = %d \n",j,i);
      a_ptr = &A_block[i*K];
      if (j == 0) copy_a (lda, K, A + i, a_ptr);
      c = C + i + j*lda;
      do_avx256_unrolled (lda, K, a_ptr, b_ptr, c);
    }
  }

  if (fringe1 != 0) {
    for ( ; i < M; ++i) {
      for (p = 0; p < N; ++p) {
        double c_ip = C[i + p*lda];
        for (int k = 0; k < K; ++k) {
	  c_ip += A[i+k*lda] * B[k+j*lda];
        }
	C[i+p*lda] = c_ip;
      }
    }  
  }
  if (fringe2 != 0) {
    Mmax = M - fringe1;
    for ( ; j < N; ++j) {
      for (i = 0; i < Mmax; ++i) {
        double c_ij = C[i + j*lda];
        for (int k = 0; k < K; ++k) {
	  c_ij += A[i+k*lda] * B[k+j*lda];
        }
	C[i+j*lda] = c_ij;
      }
    }   
  }

  //printf("M = %d\t,N = %d,K = %d, \n",M,N,K);
  //printf("Did it declare?");

  /*register char status;
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
  }*/

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
  int BLOCK1 = 256;
  int BLOCK2 = 512;
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
