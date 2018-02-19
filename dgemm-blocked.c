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

/*#define ARRAY(A,i,j) (A)[(j)*lda + (i)]

static inline void calc_4x4(int lda, int K, double* a, double* b, double* c)
{
  __m256d a0x_1x, a2x_3x, 
    bx0, bx1, bx2, bx3, 
    c00_10, c20_30, 
    c01_11, c21_31,
    c02_12, c22_32,
    c03_13, c23_33;

  double* c01_11_ptr = c + lda;
  double* c02_12_ptr = c01_11_ptr + lda;
  double* c03_13_ptr = c02_12_ptr + lda;

  c00_10 = _mm256_load_pd(c);
  c20_30 = _mm256_load_pd(c+4);
  c01_11 = _mm256_load_pd(c01_11_ptr);
  c21_31 = _mm256_load_pd(c01_11_ptr + 4);
  c02_12 = _mm256_load_pd(c02_12_ptr);
  c22_32 = _mm256_load_pd(c02_12_ptr + 4);
  c03_13 = _mm256_load_pd(c03_13_ptr);
  c23_33 = _mm256_load_pd(c03_13_ptr + 4);

  for (int x = 0; x < K; ++x) 
  {
    a0x_1x = _mm256_load_pd(a);
    a2x_3x = _mm256_load_pd(a+4);
    a += 8;

    bx0 = _mm256_load_pd(b++);
    bx1 = _mm256_load_pd(b++);
    bx2 = _mm256_load_pd(b++);
    bx3 = _mm256_load_pd(b++);

    c00_10 = _mm256_add_pd(c00_10, _mm256_mul_pd(a0x_1x, bx0));
    c20_30 = _mm256_add_pd(c20_30, _mm256_mul_pd(a2x_3x, bx0));
    c01_11 = _mm256_add_pd(c01_11, _mm256_mul_pd(a0x_1x, bx1));
    c21_31 = _mm256_add_pd(c21_31, _mm256_mul_pd(a2x_3x, bx1));
    c02_12 = _mm256_add_pd(c02_12, _mm256_mul_pd(a0x_1x, bx2));
    c22_32 = _mm256_add_pd(c22_32, _mm256_mul_pd(a2x_3x, bx2));
    c03_13 = _mm256_add_pd(c03_13, _mm256_mul_pd(a0x_1x, bx3));
    c23_33 = _mm256_add_pd(c23_33, _mm256_mul_pd(a2x_3x, bx3));
  }

  _mm256_storeu_pd(c, c00_10);
  _mm256_storeu_pd((c+4), c20_30);
  _mm256_storeu_pd(c01_11_ptr, c01_11);
  _mm256_storeu_pd((c01_11_ptr+4), c21_31);
  _mm256_storeu_pd(c02_12_ptr, c02_12);
  _mm256_storeu_pd((c02_12_ptr+4), c22_32);
  _mm256_storeu_pd(c03_13_ptr, c03_13);
  _mm256_storeu_pd((c03_13_ptr+4), c23_33);
}

static inline void copy_a (int lda, const int K, double* a_src, double* a_dest) {
  //For each 4xK block-row of A
  for (int i = 0; i < K; ++i) 
  {
    *a_dest++ = *a_src;
    *a_dest++ = *(a_src+1);
    *a_dest++ = *(a_src+2);
    *a_dest++ = *(a_src+3);
    a_src += lda;
  }
}

static inline void copy_b (int lda, const int K, double* b_src, double* b_dest) {
  double *b_ptr0, *b_ptr1, *b_ptr2, *b_ptr3;
  b_ptr0 = b_src;
  b_ptr1 = b_ptr0 + lda;
  b_ptr2 = b_ptr1 + lda;
  b_ptr3 = b_ptr2 + lda;

  for (int i = 0; i < K; ++i) 
  {
    *b_dest++ = *b_ptr0++;
    *b_dest++ = *b_ptr1++;
    *b_dest++ = *b_ptr2++;
    *b_dest++ = *b_ptr3++;
  }
}*/
/* This auxiliary subroutine performs a smaller dgemm operation
 *  C := C + A * B
 * where C is M-by-N, A is M-by-K, and B is K-by-N. */
static inline void do_block (int lda, int M, int N, int K, double* A, double* B, double* C)
{
  /*double A_block[M*K], B_block[K*N];
  double *a_ptr, *b_ptr, *c;

  const int Nmax = N-7;
  int Mmax = M-7;
  int fringe1 = M%8;
  int fringe2 = N%8;

  int i = 0, j = 0, p = 0;

  // For each column of B
  for (j = 0 ; j < Nmax; j += 8) 
  {
    b_ptr = &B_block[j*K];
    // copy and transpose B_block
    copy_b(lda, K, B + j*lda, b_ptr);
    // For each row of A
    for (i = 0; i < Mmax; i += 8) {
      a_ptr = &A_block[i*K];
      if (j == 0) copy_a(lda, K, A + i, a_ptr);
      c = C + i + j*lda;
      calc_4x4(lda, K, a_ptr, b_ptr, c);
    }
  }

  // Handle "fringes" 
  if (fringe1 != 0) 
  {
    // For each row of A
    for ( ; i < M; ++i)
      // For each column of B
      for (p = 0; p < N; ++p) 
      {
        // Compute C[i,j] 
        double c_ip = ARRAY(C,i,p);
        for (int k = 0; k < K; ++k)
          c_ip += ARRAY(A,i,k) * ARRAY(B,k,p);
        ARRAY(C,i,p) = c_ip;
      }
  }
  if (fringe2 != 0) 
  {
    Mmax = M - fringe1;
    // For each column of B 
    for ( ; j < N; ++j)
      // For each row of A 
      for (i = 0; i < Mmax; ++i) 
      {
        // Compute C[i,j] 
        double cij = ARRAY(C,i,j);
        for (int k = 0; k < K; ++k)
          cij += ARRAY(A,i,k) * ARRAY(B,k,j);
        ARRAY(C,i,j) = cij;
      }
  }*/
   // int edge1 = M % 4; int edge2 = M % 4;
  //Do math here
  //if (K % 4 == 0) {
  __m256d m0,m1,m2,m3;
  for (int i = 0; i < M; i += 4) {
    for (int j = 0; j < N; ++j) {
      m0 = _mm256_setzero_pd();  
      for (int k = 0; k < K; ++k) {
	//printf("Can you see me here\n");
	m1 = _mm256_load_pd(A+i+k*lda);
	m2 = _mm256_broadcast_sd(B+k+j*lda); // should be m2 = _mm_broadcast_pd(B+k+lda*j), 
				     // doesn't want to allow this implicit function
	m3 = _mm256_mul_pd(m1,m2);
	m0 = _mm256_add_pd(m0,m3);
      }
      _mm256_store_pd(C+i+j*lda,m0);
    }
  }
  //}
  
  /*double m0,m1,m2,m3 = 0.0;
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      m0 = C[i+j*lda];  
      for (int k = 0; k < K; ++k) {
	//printf("Can you see me here\n");
	m1 = A[i+k*lda];
	m2 = B[k+j*lda];
	m3 = m1 * m2;
	m0 = m0 + m3;
      }
      C[i+j*lda] = m0;
    }
  }*/

  /*if (edge1 != 0) {
    for (int x = 0; x < M; ++x) {
      for (int y = 0; y < N; ++y) {
	m0 = C[x+y*lda];
        for (int z = 0; z < K; ++z) {
	  m1 = A[x+z*lda];
	  m2 = B[z+y*lda];
	  m3 = m1 * m2;
	  m0 += m3;
	}
	C[x+lda*y] = m0;
      }
    }
  }  
  if (edge2 != 0) {
    for (int x = 0; x < M; ++x) {
      for (int y = 0; y < N; ++y) {
	m0 = C[x+y*lda];
        for (int z = 0; z < K; ++z) {
	  m1 = A[x+z*lda];
	  m2 = B[z+y*lda];
	  m3 = m1 * m2;
	  m0 += m3;
	}
	C[x+y*lda] = m0;
      }
    }
  } */ 
}

/* This routine performs a dgemm operation
 *  C := C + A * B
 * where A, B, and C are lda-by-lda matrices stored in column-major format. 
 * On exit, A and B maintain their input values. */
void square_dgemm (/*int iii,*/int lda, double* A, double* B, double* C)
{
  //printf("Do you see me now %d\t",lda);
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
  
  //This is for testing dgemm aglorithm outside do_block function
  /*__m128d m0,m1,m2,m3;
  for (int i = 0; i < lda; i += 8) {
    for (int j = 0; j < lda; ++j) {
      m0 = _mm_setzero_pd();  
      for (int k = 0; k < lda; ++k) {
	m1 = _mm_load_pd(A+i+lda*k);
	m2 = _mm_load_pd(B+k+lda*j);
	m3 = _mm_mul_pd(m1,m2);
	m0 = _mm_add_pd(m0,m3);
      }
      _mm_store_pd(C+i+lda*j,m0);
    }
  }*/
  
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
		//int N = min(BLOCK1,lim_j-j);
		//int K = min (BLOCK1,lim_k-k);
	      //printf("M = %d\t, N = %d, K = %d \n",M,N,K); 
              do_block(lda, M, N, K, A + i + k*lda, B + k + j*lda, C + i + j*lda);
            }
          }
        }
      }
    }
  }
 
  //Proposed blocking method for 1 level of memory - L1
  /*for (int i = 0; i < lda; i += BLOCK1) {
    int M = min (BLOCK1,lda-i);
    for (int j = 0; j < lda; j += BLOCK1) {
      int N = min (BLOCK1,lda-j);
      for (int k = 0; k < lda; k += BLOCK1) {
	//int M = min (BLOCK1,lda-i);
	//int N = min (BLOCK1,lda-j);
	int K = min (BLOCK1,lda-k);
	//printf("lad = %d\t, M = %g\t, N = %g, K = %g",lda,M,N,K);
	do_block (lda, M, N, K, A + i + k*lda, B + k + j*lda, C + i + j*lda);
      }
    }
  }*/

  //This is to print out a small n*n C matrix (after calculation)
  /*if (lda == 96) {
    for (int i = 0; i < lda; ++i) {
      for (int k = 0; k < lda; ++k) { 
	printf("%g\t ",C[i+lda*k]);
      }
      printf("\n");
    }
  }*/
}
