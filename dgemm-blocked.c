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

/*#if !defined(BLOCK_SIZiE)
#define BLOCK_SIZE 72
#endif*/

#define min(a,b) (((a)<(b))?(a):(b))

/* This auxiliary subroutine performs a smaller dgemm operation
 *  C := C + A * B
 * where C is M-by-N, A is M-by-K, and B is K-by-N. */
static void do_block (int lda, int M, int N, int K, double* A, double* B, double* C)
{    
    //register double ff[4] = {0.0,0.0,0.0,0.0};
    register double cij = 0.0;
    if ((K % 4) == 0) {
    register double ff[8];
      /* For each row i of A */
      for (int i = 0; i < M; ++i)
        /* For each column j of B */
        for (int j = 0; j < N; ++j)
        {
          /* Compute C(i,j) */
          cij = C[i+j*lda];
            for (int k = 0; k < K; k += 4)
            {
          	ff[0] = A[k+i*lda];
        	ff[1] = A[(k+1)+i*lda];
        	ff[2] = A[(k+2)+i*lda];
        	ff[3] = A[(k+3)+i*lda];
		ff[4] = B[k+j*lda];
		ff[5] = B[(k+1)+j*lda];
		ff[6] = B[(k+2)+j*lda];
		ff[7] = B[(k+3)+j*lda];
                cij += ff[0] * ff[4];
        	cij += ff[1] * ff[5];
		cij += ff[2] * ff[6];
		cij += ff[3] * ff[7];
            }
          C[i+j*lda] = cij;
        }
    }
    else if ((K % 2) != 0) {
    register double ff[4];
      /* For each row i of A */
      for (int i = 0; i < M; ++i)
        /* For each column j of B */
        for (int j = 0; j < N; ++j)
        {
          /* Compute C(i,j) */
          cij = C[i+j*lda];
	  cij += A[i*lda] * B[j*lda];
          for (int k = 1; k < K; k += 2)
          {
        	ff[0] = A[k+i*lda];
       		ff[1] = B[k+j*lda];
		ff[2] = A[(k+1)+i*lda];
		ff[3] = B[(k+1)+j*lda];
           	cij += ff[0] * ff[1];
		cij += ff[2] * ff[3];
          }
          C[i+j*lda] = cij;
        }
    }
    else {
    register double ff[4];
      /* For each row i of A */
      for (int i = 0; i < M; ++i)
        /* For each column j of B */
        for (int j = 0; j < N; ++j)
        {
          /* Compute C(i,j) */
          cij = C[i+j*lda];
            for (int k = 0; k < K; k += 2)
            {
                ff[0] = A[k+i*lda];
                ff[1] = A[(k+1)+i*lda];
                ff[2] = B[k+j*lda];
                ff[3] = B[(k+1)+j*lda];
                cij += ff[0] * ff[2];
                cij += ff[1] * ff[3];
            }
          C[i+j*lda] = cij;
        }
    }
}

/* This routine performs a dgemm operation
 *  C := C + A * B
 * where A, B, and C are lda-by-lda matrices stored in column-major format. 
 * On exit, A and B maintain their input values. */
void square_dgemm (/*int iii,*/int lda, double* A, double* B, double* C)
{
  register int BLOCK_SIZE = 72;//, M, N, K;//was 72
  int M,N,K,m = 0;
  double* AT = NULL;
  AT=(double*) malloc(lda*lda*sizeof(double));
  //Transposes A in memory for better allocation
  for (int i = 0; i < lda; ++i) {
      for (int k = 0; k < lda; ++k) {
          AT[m]=A[i+(k*lda)];
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
