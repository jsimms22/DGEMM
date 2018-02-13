#include <stdio.h>
#include <stdlib.h>
#include <malloc.h>
#include <string.h>
//#include <emmintrin.h>

const char* dgemm_desc = "Tuned blocked dgemm, based on Goto.";

#if !defined(BLOCK_SIZE)

#define BLOCK_SIZE 32

#define BLOCK_SIZE_32 32
#define BLOCK_SIZE_64 64
#define BLOCK_SIZE_128 128

// Register blocking
#define REG_BLOCK_SIZE 16
#define REG_BLOCK_ITEMS (REG_BLOCK_SIZE * REG_BLOCK_SIZE)

#endif

#define min(a,b) (((a)<(b))?(a):(b))
#define max(a,b) (((a)>(b))?(a):(b))


void print_matrix(char* header, const int lda, int M, int N, double * A)
{
	return;
	printf("\n%s\n", header);
	for(int i=0; i<M; i++) {
		printf("[");
		for(int j=0; j<N; j++) {
			printf("% .4f,", A[lda*j+i]);
		}
		printf("],\n");
	}
	printf("])\n\n");
}

/* C += A * B
 * (m x n) = (m x k) * (k x n)
 */
																 
/* Takes in column-major block A with leading dimension lda,
 * with dimensions BLOCK_SIZE * BLOCK_SIZE,
 * and column-major panel B with leading dimension ldb,
 * with dimensions BLOCK_SIZE * lda
 */
static void gebp_opt1(const int lda, const int ldb, const int blocked_lda, double* A, double*restrict B, double*restrict C)
{
	const int ldaP = BLOCK_SIZE * REG_BLOCK_SIZE;
	// Pack A into aP, and transpose to row-major order
	static double aP[BLOCK_SIZE*BLOCK_SIZE]
		__attribute__((aligned(16)));

	// NEWEST REPACK
	//
	// 1 2 5 6
	// 3_4_7_8
	// 5 6 7 8
	// 5 6 7 8
	//
	// 1 3 2 4 5 7 6 8, 5 5 6 6 7 7 8 8
	// The idea is to completely linearize a row for register blocking (m_r)

	// Select the row
	for(int i=0; i<BLOCK_SIZE / REG_BLOCK_SIZE; i++) {
		// Select the column
		double *aPptr = aP + (ldaP*i);  
		double *Aptr = A + (i*REG_BLOCK_SIZE);
		for(int j=0; j<BLOCK_SIZE; j++) {
			for(int k=0; k<REG_BLOCK_SIZE; k++) {
				aPptr[k] = Aptr[k];
				//aP[(ldaP*i) + (j*REG_BLOCK_SIZE) + k] = 
				//	A[(i*REG_BLOCK_SIZE) + (lda*j) + k];
			}
			aPptr += REG_BLOCK_SIZE;
			Aptr += lda;
		}
	}

	//print_matrix("A\n", lda, BLOCK_SIZE, BLOCK_SIZE, A);
	//print_matrix("aP\n", BLOCK_SIZE * REG_BLOCK_SIZE, BLOCK_SIZE* REG_BLOCK_SIZE, BLOCK_SIZE / REG_BLOCK_SIZE, aP);

	// aP and B are both packed now, do the math
	
	// Temporary Caux array stores, add back to C later
	// Register blocked, so REG_BLOCK_SIZE columns
	static double Caux[BLOCK_SIZE*REG_BLOCK_SIZE]
		__attribute__((aligned(16)));

	// iterate over REG_BLOCK_SIZE number columns in B and C
	for(int i=0; i<blocked_lda; i+=REG_BLOCK_SIZE) {

		// Zero out Caux
		// Trust memset, it's really efficient.
		memset(Caux, '\0', sizeof(double)*REG_BLOCK_SIZE*BLOCK_SIZE);

		// Do a dumb loop and hope the cray vectorizes it for me :/

		// iterate on reg-block rows in Caux and flattened panel-rows of A
		//double * restrict cTemp = Caux;
		for(int j=0; j<BLOCK_SIZE/REG_BLOCK_SIZE; j++) {
			// iterate on cols of B, cols of C
			double * restrict bTemp = &B[(i*BLOCK_SIZE)];
			for(int k=0; k<REG_BLOCK_SIZE; k++) {
				double * cTemp = Caux + (j*REG_BLOCK_SIZE) + (BLOCK_SIZE*k);
				// iterate down the col of B, across the row of A
				for(int l=0; l<BLOCK_SIZE; l++) {
					double bItem = B[(i*BLOCK_SIZE) + (k*BLOCK_SIZE) + l];
					// iterate on row in A (linear due to repack)
					for(int m=0; m<REG_BLOCK_SIZE; m++) {
						cTemp[m] += 
							aP[(j*ldaP) + (l*REG_BLOCK_SIZE) + m] * 
							bItem;
							//bTemp[l];
					}
				}
				bTemp += BLOCK_SIZE;
				//cTemp += BLOCK_SIZE;
			}
			//cTemp += REG_BLOCK_SIZE;
		}

		// iterate on reg-block rows in Caux and flattened panel-rows of A
		/*
		double * restrict aTemp = aP; 
		for(int j=0; j<BLOCK_SIZE/REG_BLOCK_SIZE; j++) {
			// iterate down B, across A
			for(int l=0; l<BLOCK_SIZE; l++) {
				double* restrict cTemp = Caux+(j*REG_BLOCK_SIZE); 
				// iterate across B and C, REG_BLOCK_SIZE
				double * restrict bTemp = B + (i*BLOCK_SIZE) + l;
				for(int m=0; m<REG_BLOCK_SIZE; m++) {
					// B item is fixed, iterate down cols of A and C storing values
					for(int n=0; n<REG_BLOCK_SIZE; n++) {
						//Caux[(j*REG_BLOCK_SIZE) + (m*BLOCK_SIZE) +n] += 
						//	aP[(j*ldaP) + (l*REG_BLOCK_SIZE)+n] * 
						//	B[(i*BLOCK_SIZE) + (m*BLOCK_SIZE) + l];
						cTemp[n] += 
							aTemp[n] *
							bTemp[0];
					}
					bTemp += BLOCK_SIZE;
					cTemp += BLOCK_SIZE;
				}
				aTemp += REG_BLOCK_SIZE;
			}
		}
		*/

		//print_matrix("Caux = np.matrix([", BLOCK_SIZE, BLOCK_SIZE, REG_BLOCK_SIZE, Caux);
		
		// Store Caux back to proper column of C
		// Vectorized version was tested to be slightly faster
		for(int j=0; j<REG_BLOCK_SIZE; j++) {
			for(int k=0; k<BLOCK_SIZE; k++) {
				/*
				__m128d orig = _mm_loadu_pd(&C[((i+j)*lda)+k]);
				__m128d temp = _mm_load_pd(&Caux[j*BLOCK_SIZE+k]);
				temp = _mm_add_pd(temp, orig);
				_mm_storeu_pd(&C[((i+j)*lda)+k], temp);
				*/
				C[((i+j)*lda)+k] += Caux[j*BLOCK_SIZE+k];
			}
		}
	}

	//print_matrix("A = np.matrix([", lda, BLOCK_SIZE, BLOCK_SIZE, A);
	//print_matrix("B = np.matrix([", ldb, ldb, lda, B);
	//print_matrix("C = np.matrix([", lda, ldb, lda, C);

}
static void gebp_opt2(const int lda, const int ldb, const int blocked_lda, double* A, double*restrict B, double*restrict C)
{
	const int ldaP = BLOCK_SIZE_64 * REG_BLOCK_SIZE;
	static double aP[BLOCK_SIZE_64*BLOCK_SIZE_64]
		__attribute__((aligned(16)));
	for(int i=0; i<BLOCK_SIZE_64 / REG_BLOCK_SIZE; i++) {
		for(int j=0; j<BLOCK_SIZE_64; j++) {
			for(int k=0; k<REG_BLOCK_SIZE; k++) {
				aP[(ldaP*i) + (j*REG_BLOCK_SIZE) + k] = 
					A[(i*REG_BLOCK_SIZE) + (lda*j) + k];
			}
		}
	}
	static double Caux[BLOCK_SIZE_64*REG_BLOCK_SIZE]
		__attribute__((aligned(16)));
	for(int i=0; i<blocked_lda; i+=REG_BLOCK_SIZE) {
		memset(Caux, '\0', sizeof(double)*REG_BLOCK_SIZE*BLOCK_SIZE_64);
		for(int j=0; j<BLOCK_SIZE_64/REG_BLOCK_SIZE; j++) {
			for(int k=0; k<REG_BLOCK_SIZE; k++) {
				double * cTemp = Caux + (j*REG_BLOCK_SIZE) + (BLOCK_SIZE_64*k);
				for(int l=0; l<BLOCK_SIZE_64; l++) {
					double bItem = B[(i*BLOCK_SIZE_64) + (k*BLOCK_SIZE_64) + l];
					for(int m=0; m<REG_BLOCK_SIZE; m++) {
						cTemp[m] += 
							aP[(j*ldaP) + (l*REG_BLOCK_SIZE) + m] * 
							bItem;
					}
				}
			}
		}
		for(int j=0; j<REG_BLOCK_SIZE; j++) {
			for(int k=0; k<BLOCK_SIZE_64; k++) {
				C[((i+j)*lda)+k] += Caux[j*BLOCK_SIZE_64+k];
			}
		}
	}
}
// 128
static void gebp_opt3(const int lda, const int ldb, const int blocked_lda, double* A, double*restrict B, double*restrict C)
{
	const int ldaP = BLOCK_SIZE_128 * REG_BLOCK_SIZE;
	static double aP[BLOCK_SIZE_128*BLOCK_SIZE_128]
		__attribute__((aligned(16)));
	for(int i=0; i<BLOCK_SIZE_128 / REG_BLOCK_SIZE; i++) {
		for(int j=0; j<BLOCK_SIZE_128; j++) {
			for(int k=0; k<REG_BLOCK_SIZE; k++) {
				aP[(ldaP*i) + (j*REG_BLOCK_SIZE) + k] = 
					A[(i*REG_BLOCK_SIZE) + (lda*j) + k];
			}
		}
	}
	static double Caux[BLOCK_SIZE_128*REG_BLOCK_SIZE]
		__attribute__((aligned(16)));
	for(int i=0; i<blocked_lda; i+=REG_BLOCK_SIZE) {
		memset(Caux, '\0', sizeof(double)*REG_BLOCK_SIZE*BLOCK_SIZE_128);
		for(int j=0; j<BLOCK_SIZE_128/REG_BLOCK_SIZE; j++) {
			for(int k=0; k<REG_BLOCK_SIZE; k++) {
				double * cTemp = Caux + (j*REG_BLOCK_SIZE) + (BLOCK_SIZE_128*k);
				for(int l=0; l<BLOCK_SIZE_128; l++) {
					double bItem = B[(i*BLOCK_SIZE_128) + (k*BLOCK_SIZE_128) + l];
					for(int m=0; m<REG_BLOCK_SIZE; m++) {
						cTemp[m] += 
							aP[(j*ldaP) + (l*REG_BLOCK_SIZE) + m] * 
							bItem;
					}
				}
			}
		}
		for(int j=0; j<REG_BLOCK_SIZE; j++) {
			for(int k=0; k<BLOCK_SIZE_128; k++) {
				C[((i+j)*lda)+k] += Caux[j*BLOCK_SIZE_128+k];
			}
		}
	}
}

/**
 * A is in column-major order with leading dimension lda
 * A dimensions are lda * BLOCK_SIZE
 * B is in column-major order with leading dimension lda, K rows
 * B dimensions are BLOCK_SIZE * lda
 */

// Optimized for BLOCK_SIZE_32
static void gepp_blk_var1(const int lda, const int blocked_lda, double*restrict bP, double* A, double* B, double *restrict C)
{
	// Pack B into column-major bP with ldb = BLOCK_SIZE_32
	const int ldb = BLOCK_SIZE_32;
	for(int i=0; i<blocked_lda; i++) {
		for(int j=0; j<BLOCK_SIZE_32; j++) {
			bP[i*ldb+j] = B[i*lda+j];
		}
	}

	// Do gebp on each block in A and packed B
	for(int i=0; i<blocked_lda; i+=BLOCK_SIZE_32) {
		gebp_opt1(lda, ldb, blocked_lda, A+i, bP, C+i);
	}
}

// Optimized for BLOCK_SIZE_64
static void gepp_blk_var2(const int lda, const int blocked_lda, double*restrict bP, double* A, double* B, double *restrict C)
{
	// Pack B into column-major bP with ldb = BLOCK_SIZE_64
	const int ldb = BLOCK_SIZE_64;
	for(int i=0; i<blocked_lda; i++) {
		for(int j=0; j<BLOCK_SIZE_64; j++) {
			bP[i*ldb+j] = B[i*lda+j];
		}
	}

	// Do gebp on each block in A and packed B
	for(int i=0; i<blocked_lda; i+=BLOCK_SIZE_64) {
		gebp_opt2(lda, ldb, blocked_lda, A+i, bP, C+i);
	}
}

// Optimized for BLOCK_SIZE_128
static void gepp_blk_var3(const int lda, const int blocked_lda, double*restrict bP, double* A, double* B, double *restrict C)
{
	// Pack B into column-major bP with ldb = BLOCK_SIZE_128
	const int ldb = BLOCK_SIZE_128;
	for(int i=0; i<blocked_lda; i++) {
		for(int j=0; j<BLOCK_SIZE_128; j++) {
			bP[i*ldb+j] = B[i*lda+j];
		}
	}

	// Do gebp on each block in A and packed B
	for(int i=0; i<blocked_lda; i+=BLOCK_SIZE_128) {
		gebp_opt3(lda, ldb, blocked_lda, A+i, bP, C+i);
	}
}

static void do_block_transpose(const int lda, const int ldb, const int ldc, int M, int K, int N, double* A, double* B, double*restrict C)
{
	// Down cols of C
	for (int i=0; i<M; i++) {
		// Across row of C
		for(int j=0; j<N; j++) {
			// Dot product of row of A, col of B
			double cij = C[i+j*ldc];
			for(int k=0; k<K; k++) {
				cij += A[i*lda + k] * B[j*ldb + k];
			}
			C[i+j*ldc] = cij;
		}
	}
}

static void naive_all_purpose(const int lda, const int ldb, const int ldc, const int m, const int r, const int n,
		double*restrict A, double* B, double*restrict C) {
	static double aT[BLOCK_SIZE*BLOCK_SIZE];
	// Block down cols of A
	//printf("m=%d, r=%d, n=%d\n", m, r, n);
	for(int i=0; i<m; i+=BLOCK_SIZE) {
		// # rows of A block
		int M = min(BLOCK_SIZE, m-i);
		// Block across rows of A
		for(int j=0; j<r; j+=BLOCK_SIZE) {
			// # cols of A block
			int R = min(BLOCK_SIZE, r-j);

			//printf("M: %d, R: %d", M, R);
			//print_matrix("Ablock", lda, M, R, A + i + (j*lda));
			// Do the transpose
			for(int k=0; k<M; k++) {
				for(int l=0; l<R; l++) {
					aT[(k*R)+l] = A[i + (j*lda) + (l*lda) + k];
					//printf("aT: %d, A: %d\n", (k*R) + l, (l*lda) + k);
				}
			}
			//print_matrix("AT", R, R, M, aT);

			// Multiply aT with each block in row of B across B/C
			for(int k=0; k<n; k+= BLOCK_SIZE) {
				int K = min(BLOCK_SIZE, n-k);
				do_block_transpose(R, ldb, ldc, 
						M, R, K,
						aT, B+j+k*lda, C+i+(k*lda));
			}
		}
	}
}

/*
static void naive_all_purpose(const int lda, const int ldb, const int ldc, const int m, const int r, const int n,
		double* A, double* B, double*restrict C) {
	// Row of A
	for(int i=0; i<m; i++) {
		// Col of B
		for(int j=0; j<n; j++) {
			double cij = C[i+(j*ldc)];
			// Across row of A, down col of B
			for(int k=0; k<r; k++) {
				cij += A[i + (k*lda)] * B[(j*ldb) + k];
			}
			C[i + (j*ldc)] = cij;
		}
	}
}
*/

/* This routine performs a dgemm operation
 *  C := C + A * B
 * where A, B, and C are lda-by-lda matrices stored in column-major format. 
 * On exit, A and B maintain their input values. */  
void square_dgemm (int lda, double* A, double* B, double*restrict C)
{
	//print_matrix("A = np.matrix([", lda, lda, lda, A);
	//print_matrix("B = np.matrix([", lda, lda, lda, B);
	//print_matrix("C = np.matrix([", lda, lda, lda, C);
  //naive_all_purpose(
  //		lda, lda, lda, 
	//		lda, lda, lda,
	//		A, B, C);
	//return;
	// Block out into a panel x panel, and call gepp
	// A iterates right by columns
	
	int blocked_lda = lda;
	int fringe = 0;
	
	if(lda % BLOCK_SIZE != 0) {
		// Order by commonality, not like a branch here or there really matters.
		int current = 0;
		while(current < lda) {
			if(current + BLOCK_SIZE > lda) {
				break;
			}
			else {
				current += BLOCK_SIZE;
			}
		}
		// Dimensions of the big block
		blocked_lda = current;
		// Size of the fringe
		fringe = lda - blocked_lda;
	}
	//printf("Size: %d, Blocked: %d, Fringe: %d\n", lda, blocked_lda, fringe);

	// First do the big block

	// First two are disabled for now
	if(blocked_lda % BLOCK_SIZE_128 == 0) {
		double * restrict bP = memalign(16, sizeof(double)*blocked_lda*BLOCK_SIZE_128);
  	for (int i = 0; i < blocked_lda; i += BLOCK_SIZE_128) {
				gepp_blk_var3(lda, blocked_lda, bP, A + (i*lda), B + i, C);
		}
		free(bP);
  }
	else if(blocked_lda % BLOCK_SIZE_64 == 0) {
		double * restrict bP = memalign(16, sizeof(double)*blocked_lda*BLOCK_SIZE_64);
  	for (int i = 0; i < blocked_lda; i += BLOCK_SIZE_64) {
				gepp_blk_var2(lda, blocked_lda, bP, A + (i*lda), B + i, C);
		}
		free(bP);
  }
	else if(blocked_lda % BLOCK_SIZE_32 == 0) {
		double * restrict bP = memalign(16, sizeof(double)*blocked_lda*BLOCK_SIZE_32);
  	for (int i = 0; i < blocked_lda; i += BLOCK_SIZE_32) {
				gepp_blk_var1(lda, blocked_lda, bP, A + (i*lda), B + i, C);
		}
		free(bP);
  }

  // Do the fringes
  if(fringe > 0) {
		//print_matrix("C = np.matrix([", lda, lda, lda, C);
		
		int tall_skinny_off = lda * blocked_lda;
		int short_fat_off = blocked_lda;
		int small_block_off = tall_skinny_off + short_fat_off;

		// Big block of C needs tall skinny from A, wide long from B
  	naive_all_purpose(
  			lda, lda, lda, 
  			blocked_lda, fringe, blocked_lda, 
				A+tall_skinny_off, B+short_fat_off, C);
		//printf("Big block\n");
		//print_matrix("C = np.matrix([", lda, lda, lda, C);
		// Tall skinny block of C needs (big block from A * tall skinny from B)
  	naive_all_purpose(
  			lda, lda, lda, 
  			blocked_lda, blocked_lda, fringe, 
				A, B+tall_skinny_off, C+tall_skinny_off);
		// Tall skinny block of C needs (tall skinny from A * small block from B)
  	naive_all_purpose(
  			lda, lda, lda, 
  			blocked_lda, fringe, fringe, 
				A+tall_skinny_off, B+small_block_off, C+tall_skinny_off);
		//printf("Tall skinny\n");
		//print_matrix("C = np.matrix([", lda, lda, lda, C);
		// Short fat block from C needs (short fat from A * big block from B)
		naive_all_purpose(
				lda, lda, lda,
				fringe, blocked_lda, blocked_lda,
				A+short_fat_off, B, C+short_fat_off);
		// Short fat block from C needs (small block from A * short fat from B)
		naive_all_purpose(
				lda, lda, lda,
				fringe, fringe, blocked_lda,
				A+small_block_off, B+short_fat_off, C+short_fat_off);
		//printf("Short fat\n");
		//print_matrix("C = np.matrix([", lda, lda, lda, C);
		// Small block from C needs (short fat from A * tall skinny from B)
		naive_all_purpose(
				lda, lda, lda,
				fringe, blocked_lda, fringe,
				A+short_fat_off, B+tall_skinny_off, C+small_block_off);
		// Small block from C needs (small from A * small from B)
		naive_all_purpose(
				lda, lda, lda,
				fringe, fringe, fringe,
				A+small_block_off, B+small_block_off, C+small_block_off);
		//printf("Done C\n");
		//print_matrix("C = np.matrix([", lda, lda, lda, C);
  }
}
