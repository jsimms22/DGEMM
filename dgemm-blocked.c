#include <mmintrin.h>
#include <immintrin.h>
#include <xmmintrin.h>
#include <pmmintrin.h>
#include <emmintrin.h>
#include <string.h>

const char* dgemm_desc = "Simple blocked dgemm.";

//#define BLOCK1 256
//#define BLOCK2 512

#define turn_even(x) (((x) & 1) ? (x+1) : (x))
#define min(a,b) (((a)<(b))?(a):(b))

#define likely(x)       __builtin_expect((x),1)
#define unlikely(x)     __builtin_expect((x),0)

#define ARRAY(A,i,j) (A)[(j)*lda + (i)]

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
static void do_8x8 (int lda, int K, double* a, double* b, double* c) {
  __m256d a0x_3x, a4x_7x,
    bx0, bx1, bx2, bx3, 
    bx4, bx5, bx6, bx7,
    c00_30, c40_70, 
    c01_31, c41_71,
    c02_32, c42_72, 
    c03_33, c43_73,
    c04_34, c44_74, 
    c05_35, c45_75,
    c06_36, c46_76,
    c07_37, c47_77;
  
  double* c01_31_ptr = c + lda;
  double* c02_32_ptr = c01_31_ptr + lda;
  double* c03_33_ptr = c02_32_ptr + lda;
  double* c04_34_ptr = c03_33_ptr + lda;
  double* c05_35_ptr = c04_34_ptr + lda;
  double* c06_36_ptr = c05_35_ptr + lda;
  double* c07_37_ptr = c06_36_ptr + lda;

  c00_30 = _mm256_loadu_pd(c);
  c40_70 = _mm256_loadu_pd(c + 4);
  c01_31 = _mm256_loadu_pd(c01_31_ptr);
  c41_71 = _mm256_loadu_pd(c01_31_ptr + 4);
  c02_32 = _mm256_loadu_pd(c02_32_ptr);
  c42_72 = _mm256_loadu_pd(c02_32_ptr + 4);
  c03_33 = _mm256_loadu_pd(c03_33_ptr);
  c43_73 = _mm256_loadu_pd(c03_33_ptr + 4);
  c04_34 = _mm256_loadu_pd(c04_34_ptr);
  c44_74 = _mm256_loadu_pd(c04_34_ptr + 4);
  c05_35 = _mm256_loadu_pd(c05_35_ptr);
  c45_75 = _mm256_loadu_pd(c05_35_ptr + 4);
  c06_36 = _mm256_loadu_pd(c06_36_ptr);
  c46_76 = _mm256_loadu_pd(c06_36_ptr + 4);
  c07_37 = _mm256_loadu_pd(c07_37_ptr);
  c47_77 = _mm256_loadu_pd(c07_37_ptr + 4);
  
  for (int x = 0; x < K; ++x) {
    a0x_3x = _mm256_loadu_pd(a);
    a4x_7x = _mm256_loadu_pd(a+4);
    a += 8;

    bx0 = _mm256_broadcast_sd(b++);
    bx1 = _mm256_broadcast_sd(b++);
    bx2 = _mm256_broadcast_sd(b++);
    bx3 = _mm256_broadcast_sd(b++);
    bx4 = _mm256_broadcast_sd(b++);
    bx5 = _mm256_broadcast_sd(b++);
    bx6 = _mm256_broadcast_sd(b++);
    bx7 = _mm256_broadcast_sd(b++);

    c00_30 = _mm256_add_pd(c00_30, _mm256_mul_pd(a0x_3x,bx0));
    c40_70 = _mm256_add_pd(c40_70, _mm256_mul_pd(a4x_7x,bx0));
    c01_31 = _mm256_add_pd(c01_31, _mm256_mul_pd(a0x_3x,bx1));
    c41_71 = _mm256_add_pd(c41_71, _mm256_mul_pd(a4x_7x,bx1));
    c02_32 = _mm256_add_pd(c02_32, _mm256_mul_pd(a0x_3x,bx2));
    c42_72 = _mm256_add_pd(c42_72, _mm256_mul_pd(a4x_7x,bx2));
    c03_33 = _mm256_add_pd(c03_33, _mm256_mul_pd(a0x_3x,bx3));
    c43_73 = _mm256_add_pd(c43_73, _mm256_mul_pd(a4x_7x,bx3));
    c04_34 = _mm256_add_pd(c04_34, _mm256_mul_pd(a0x_3x,bx4));
    c44_74 = _mm256_add_pd(c44_74, _mm256_mul_pd(a4x_7x,bx4));
    c05_35 = _mm256_add_pd(c05_35, _mm256_mul_pd(a0x_3x,bx5));
    c45_75 = _mm256_add_pd(c45_75, _mm256_mul_pd(a4x_7x,bx5));
    c06_36 = _mm256_add_pd(c06_36, _mm256_mul_pd(a0x_3x,bx6));
    c46_76 = _mm256_add_pd(c46_76, _mm256_mul_pd(a4x_7x,bx6));
    c07_37 = _mm256_add_pd(c07_37, _mm256_mul_pd(a0x_3x,bx7));
    c47_77 = _mm256_add_pd(c47_77, _mm256_mul_pd(a4x_7x,bx7));
  }

  _mm256_storeu_pd(c,c00_30);
  _mm256_storeu_pd(c+4,c40_70);
  _mm256_storeu_pd(c01_31_ptr,c01_31);
  _mm256_storeu_pd(c01_31_ptr+4,c41_71);
  _mm256_storeu_pd(c02_32_ptr,c02_32);
  _mm256_storeu_pd(c02_32_ptr+4,c42_72);
  _mm256_storeu_pd(c03_33_ptr,c03_33);
  _mm256_storeu_pd(c03_33_ptr+4,c43_73);
  _mm256_storeu_pd(c04_34_ptr,c04_34);
  _mm256_storeu_pd(c04_34_ptr+4,c44_74);
  _mm256_storeu_pd(c05_35_ptr,c05_35);
  _mm256_storeu_pd(c05_35_ptr+4,c45_75);
  _mm256_storeu_pd(c06_36_ptr,c06_36);
  _mm256_storeu_pd(c06_36_ptr+4,c46_76);
  _mm256_storeu_pd(c07_37_ptr,c07_37);
  _mm256_storeu_pd(c07_37_ptr+4,c47_77);
}

static inline void copy_a8 (int lda, const int K, double* a_src, double* a_dest) {
  for (int i = 0; i < K; ++i) {
    *a_dest++ = *a_src;
    *a_dest++ = *(a_src + 1);
    *a_dest++ = *(a_src + 2);
    *a_dest++ = *(a_src + 3);
    *a_dest++ = *(a_src + 4);
    *a_dest++ = *(a_src + 5);
    *a_dest++ = *(a_src + 6);
    *a_dest++ = *(a_src + 7);
    a_src += lda;
  }
}

static inline void copy_b8 (int lda, const int K, double* b_src, double* b_dest) {
  double *b_ptr0, *b_ptr1, *b_ptr2, *b_ptr3,
  *b_ptr4, *b_ptr5, *b_ptr6, *b_ptr7;
  b_ptr0 = b_src;
  b_ptr1 = b_ptr0 + lda;
  b_ptr2 = b_ptr1 + lda;
  b_ptr3 = b_ptr2 + lda;
  b_ptr4 = b_ptr3 + lda;
  b_ptr5 = b_ptr4 + lda;
  b_ptr6 = b_ptr5 + lda;
  b_ptr7 = b_ptr6 + lda;

  for (int i = 0; i < K; ++i) {
    *b_dest++ = *b_ptr0++;
    *b_dest++ = *b_ptr1++;
    *b_dest++ = *b_ptr2++;
    *b_dest++ = *b_ptr3++;
    *b_dest++ = *b_ptr4++;
    *b_dest++ = *b_ptr5++;
    *b_dest++ = *b_ptr6++;
    *b_dest++ = *b_ptr7++;
  }
}

 /*C Matrix 4x4     A Matrix   B Matrix
 * | 00 10 20 30 |  | 0x -> |  | 0x 1x 2x 3x |
 * | 01 11 21 31 |  | 1x -> |  |             |
 * | 02 12 22 32 |  | 2x -> |  |             |
 * | 03 13 23 33 |  | 3x -> |  |             |
 */
static void do_4x4 (int lda, int K, double* a, double* b, double* c) {
  register __m256d a0x_3x,
    bx0, bx1, bx2, bx3,
    c00_30, c01_31,
    c02_32, c03_33;
  
  double* c01_31_ptr = c + lda;
  double* c02_32_ptr = c01_31_ptr + lda;
  double* c03_33_ptr = c02_32_ptr + lda;
  
  c00_30 = _mm256_loadu_pd(c);
  c01_31 = _mm256_loadu_pd(c01_31_ptr);
  c02_32 = _mm256_loadu_pd(c02_32_ptr);
  c03_33 = _mm256_loadu_pd(c03_33_ptr);
  
  for (int x = 0; x < K; ++x) {
    a0x_3x = _mm256_loadu_pd(a);
    a += 4;

    bx0 = _mm256_broadcast_sd(b++);
    bx1 = _mm256_broadcast_sd(b++);
    bx2 = _mm256_broadcast_sd(b++);
    bx3 = _mm256_broadcast_sd(b++);

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

static inline void copy_a4 (int lda, const int K, double* a_src, double* a_dest) {
  for (int i = 0; i < K; ++i) {
    *a_dest++ = *a_src;
    *a_dest++ = *(a_src + 1);
    *a_dest++ = *(a_src + 2);
    *a_dest++ = *(a_src + 3);
    a_src += lda;
  }
}

static inline void copy_b4 (int lda, const int K, double* b_src, double* b_dest) {
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

 /*C Matrix 2x2 A Matrix   B Matrix
 * | 00 10 |  | 0x -> |  | 0x 1x |
 * | 01 11 |  | 1x -> |  |       |
 */
static void do_2x2 (int lda, int K, double* a, double* b, double* c) {
  register __m128d a0x_1x,
    bx0, bx1,
    c00_10,
    c01_11;
  
  double* c01_11_ptr = c + lda;
  
  c00_10 = _mm_loadu_pd(c);
  c01_11 = _mm_loadu_pd(c01_11_ptr);
  
  for (int x = 0; x < K; ++x) {
    a0x_1x = _mm_load_pd(a);
    a += 2;

    bx0 = _mm_loaddup_pd(b++);
    bx1 = _mm_loaddup_pd(b++);

    c00_10 = _mm_add_pd(c00_10, _mm_mul_pd(a0x_1x,bx0));
    c01_11 = _mm_add_pd(c01_11, _mm_mul_pd(a0x_1x,bx1));
  }
  
  _mm_storeu_pd(c,c00_10);
  _mm_storeu_pd(c01_11_ptr,c01_11);
}

static inline void copy_a2 (int lda, const int K, double* a_src, double* a_dest) {
  for (int i = 0; i < K; ++i) {
    *a_dest++ = *a_src;
    *a_dest++ = *(a_src + 1);
    a_src += lda;
  }
}

static inline void copy_b2 (int lda, const int K, double* b_src, double* b_dest) {
  double *b_ptr0, *b_ptr1;
  b_ptr0 = b_src;
  b_ptr1 = b_ptr0 + lda;

  for (int i = 0; i < K; ++i) {
    *b_dest++ = *b_ptr0++;
    *b_dest++ = *b_ptr1++;
  }
}

static void do_avx256 (int lda, int M, int N, int K, double* a, double* b, double* c) {
  __m256d m0,m1,m2,m3;
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
static void inline do_block (int lda, int M, int N, int K, double* A, double* B, double* C)
{
  double A_block[M*K], B_block[K*N];
  double *a_ptr, *b_ptr, *c;

/*
 * 8x8 blocks: slower than 4x4 blocks
 */
  /*int Nmax = N-7;
  int Mmax = M-7;
  int fringe1 = M%8;
  int fringe2 = N%8;
  
  int i = 0, j = 0, p = 0;
    
  for (j = 0; j < Nmax; j += 8) {
    b_ptr = &B_block[j*K];
    copy_b8 (lda, K, B + j*lda, b_ptr);
    for (i = 0; i < Mmax; i += 8) {
      a_ptr = &A_block[i*K];
      if (j == 0) copy_a8 (lda, K, A + i, a_ptr);
      c = C + i + j*lda;
      do_8x8 (lda, K, a_ptr, b_ptr, c);
    }
  }*/

 /* 4x4 blocks */
  int Nmax = N-3;
  int Mmax = M-3;
  int fringe1 = M%4;
  int fringe2 = N%4;

  register int i = 0, j = 0, p = 0;

  for (j = 0 ; j < Nmax; j += 4) 
  {
    b_ptr = &B_block[j*K];
    copy_b4(lda, K, B + j*lda, b_ptr);
    for (i = 0; i < Mmax; i += 4) {
      a_ptr = &A_block[i*K];
      if (j == 0) copy_a4(lda, K, A + i, a_ptr);
      c = C + i + j*lda;
      do_4x4(lda, K, a_ptr, b_ptr, c);
    }
  }

 /* 2x2 blocks */
  /*int Nmax = N-1;
  int Mmax = M-1;
  int fringe1 = M%2;
  int fringe2 = N%2;
  
  int i = 0, j = 0, p = 0;
  
  for (j = 0; j < Nmax; j += 2) {
    b_ptr = &B_block[j*K];
    copy_b2 (lda, K, B + j*lda, b_ptr);
    for (i = 0; i < Mmax; i += 2) {
      a_ptr = &A_block[i*K];
      if (j == 0) copy_a2 (lda, K, A + i, a_ptr);
      c = C + i + j*lda;
      do_2x2 (lda, K, a_ptr, b_ptr, c);
    }
  }*/

  /* Handle "fringes" */
  if (fringe1 != 0) {
    /* For each row of A */
    for ( ; i < M; ++i)
      /* For each column of B */ 
      for (p = 0; p < N; ++p) {
        /* Compute C[i,j] */
        register double c_ip = ARRAY(C,i,p);
        for (int k = 0; k < K; ++k)
          c_ip += ARRAY(A,i,k) * ARRAY(B,k,p);
        ARRAY(C,i,p) = c_ip;
      }
  }

  if (fringe2 != 0) {
    Mmax = M - fringe1;
    /* For each column of B */
    for ( ; j < N; ++j)
      /* For each row of A */ 
      for (i = 0; i < Mmax; ++i) {
        /* Compute C[i,j] */
        register double cij = ARRAY(C,i,j);
        for (int k = 0; k < K; ++k)
          cij += ARRAY(A,i,k) * ARRAY(B,k,j);
        ARRAY(C,i,j) = cij;
      }
  }
}

/* This routine performs a dgemm operation
 *  C := C + A * B
 * where A, B, and C are lda-by-lda matrices stored in column-major format. 
 * On exit, A and B maintain their input values. */
void square_dgemm (/*int iii,*/ int lda, double* A, double* B, double* C)
{
  //Block size in 1D for L2 cache - BLOCK2 - for L1 cache - BLOCK 1 - 
  register int BLOCK1 = 256;
  register int BLOCK2 = 512;

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
              do_block(lda, M, N, K, A + i + k*lda, B + k + j*lda, C + i + j*lda);
            }
          }
        }
      }
    }
  }
}
