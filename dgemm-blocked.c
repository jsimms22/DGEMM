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
const char* dgemm_desc = "Simple blocked dgemm.";

#define min(a,b) (((a)<(b))?(a):(b))

/* This auxiliary subroutine performs a smaller dgemm operation
 *  C := C + A * B
 * where C is M-by-N, A is M-by-K, and B is K-by-N. */
static void do_block (int lda, int M, int N, int K, double* A, double* B, double* C)
{
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      C[i*lda+j] = 0.0;
      for (int k = 0; k < K; ++k) {
	C[i*lda+j] += A[i*lda+k] * B[k*lda+j];
      }
    }
  }   	
}

/* This routine performs a dgemm operation
 *  C := C + A * B
 * where A, B, and C are lda-by-lda matrices stored in column-major format. 
 * On exit, A and B maintain their input values. */
void square_dgemm (/*int iii,*/int lda, double* A, double* B, double* C)
{
  register int BLOCK_SIZE = 72;
  register int m = 0;
  int M,N,K = 0;

  double* AT = NULL;
  AT=(double*) malloc(lda*lda*sizeof(double));  
  //Transposes A in memory for better allocation
  for (int i = 0; i < lda; ++i) {
    for (int k = 0; k < lda; ++k) {
      AT[m] = A[i+(k*lda)];
      BT[m] = B[i+(k*lda)];
      ++m;
    }
  }

  /* For each block-row of A */
  for (int i = 0; i < lda; i += BLOCK_SIZE)
    /* For each block-column of B */
    for (int j = 0; j < lda; j += BLOCK_SIZE)
      /* Accumulate block dgemms into block of C */
      for (int k = 0; k < lda; k += BLOCK_SIZE){
	/* Correct block dimensions if block "goes off edge of" the matrix */
	M = min (BLOCK_SIZE, lda-i);
	N = min (BLOCK_SIZE, lda-j);
	K = min (BLOCK_SIZE, lda-k);
	/*Perform individual block dgemm*/
        do_block(lda, M, N, K, AT + k + i*lda, B + k + j*lda, C + i + j*lda);
      }
  free(AT);
}
