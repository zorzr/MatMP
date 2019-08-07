#include <omp.h>
#include <cstdio>
#include <utility>
#include "inverse.hpp"
using namespace std;


//-------------------------------------- SERIAL IMPLEMENTATION------------------------------------------

// Partial pivoting on P and on PA (copy of matrix A to be factorized) 
double* pivoting(double* A, double* PA, unsigned int n) {
    double* P = new double[n*n];
    unsigned int i, j, k;

    // Initialization
    for (i = 0; i < n; i++) {
        P[i*(n+1)] = 1;
		for (j = 0; j < n; j++)
			PA[i*n+j] = A[i*n+j];
	}

    // Pivoting observing the maximum absolute value
    for (k = 0; k < n-1; k++) {
        double max = 0;
        for (i = k; i < n; i++) {
            unsigned long v = i*n+k;
            double p = A[v]*A[v];
            if (p > max) {
                j = i;
                max = p;
            }
        }

        for (i = 0; i < n; i++) {
            swap(P[k*n+i], P[j*n+i]);
            swap(PA[k*n+i], PA[j*n+i]);
		}
    }
    
    return P;
}


// Serial inverse computation using LU factorization
double* lu_serial(double* A, unsigned int n) {
	unsigned int i, j, k, l;
    unsigned long in, jn;

    // Partial pivoting
    double* PA = new double[n*n];
    double* P = pivoting(A,PA,n);

    // LU factorization inside a single matrix PA (lower side L and upper side U)
    for (i = 0; i < n; i++) {
        in = i*n;
        for (j = i+1; j < n; j++) {
            jn = j*n;
            PA[jn+i] /= PA[in+i];
            for (k = i+1; k < n; k++) {
                PA[jn+k] -= PA[jn+i] * PA[in+k];
            }
        }
    }
    
    double* iA = new double[n*n];
    double sum;

    // Inverse computation column by column
	for (j = 0; j < n; j++) {
		double* x = new double[n]();
		double* y = new double[n]();

        // Lower triangular solve
		y[0] = P[j];
		for (i = 1; i < n; i++) {
			sum = 0;
			for (l = 0; l < i; l++)
				sum += PA[i*n+l] * y[l];
			y[i] = P[i*n+j] - sum;
		}

        // Upper triangular solve
		x[n-1] = y[n-1] / PA[(n-1)*n + n-1];
		for (i = n-2; i < n; i--) {
			sum = 0;
			for (l = i+1; l < n; l++)
				sum += PA[i*n+l]*x[l];
			x[i] = (y[i] - sum) / PA[i*n+i];
		}

		for (i = 0; i < n; i++)
			iA[i*n+j] = x[i];
		
		delete[] x;
		delete[] y;
	}

    delete[] P;
    delete[] PA;
	return iA;
}



// -------------------------------------------PARALLEL IMPLEMENTATIONS-------------------------------------------

// Parallel partial pivoting on P and PA
double* pivoting_p(double* A, double* PA, unsigned int n) {
    double* P = new double[n*n];
    unsigned int i, j, k;

    // Parallelized initialization (cheaper for huge n)
    #pragma omp parallel for collapse(2)
    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            if (i == j)  P[i*(n+1)] = 1;
            PA[i*n+j] = A[i*n+j];
        }
    }
   
    // Hardly parallelizable, not worth for n big
    for (k = 0; k < n-1; k++) {
        double max = 0;
        for (i = k; i < n; i++) {
            unsigned long v = i*n+k;
            double p = A[v]*A[v];
            if (p > max) {
                j = i;
                max = p;
            }
        }
        
        for (i = 0; i < n; i++) {
            swap(P[k*n+i], P[j*n+i]);
            swap(PA[k*n+i], PA[j*n+i]);
        }
    }
    
    return P;
}


// Parallel LU inverse computation (simple, we work on for loops)
double* lu_naive(double* A, unsigned int n) {
    double* iA = new double[n*n];
    double* PA = new double[n*n];
    double* P = pivoting_p(A,PA,n);
    
    #pragma omp parallel
	{
        // Inner loop (j) depends from the outer (i), we have to parallelize the inner
        for (unsigned int i = 0; i < n; i++) {
            unsigned long in = i*n;
			#pragma omp for schedule(dynamic, 4)
			for (unsigned int j = i+1; j < n; j++) {
                unsigned long jn = j*n;
				PA[jn+i] /= PA[in+i];
				for (unsigned int k = i+1; k < n; k++) {
					PA[jn+k] -= PA[jn+i] * PA[in+k];
				}
			}
		}

		// Each loop works on a different column: can be paralleized
		#pragma omp for schedule(dynamic)
		for (unsigned int j = 0; j < n; j++) {
			double sum;
			unsigned int i, l;
			double* x = new double[n]();
			double* y = new double[n]();

            // Lower triangular solve
			y[0] = P[j];
			for (i = 1; i < n; i++) {
				sum = 0;
				for (l = 0; l < i; l++)
					sum += PA[i*n+l] * y[l];
				y[i] = P[i*n+j] - sum;
			}

            // Upper triangular solve
			x[n-1] = y[n-1] / PA[n*n-1];
			for (i = n-2; i < n; i--) {
				sum = 0;
				for (l = i+1; l < n; l++)
					sum += PA[i*n+l] * x[l];
				x[i] = (y[i] - sum) / PA[i*n+i];
			}
			
			for (i = 0; i < n; i++)
				iA[i*n+j] = x[i];

			delete[] x;
			delete[] y;
		}
	}

    delete[] P;
    delete[] PA;
	return iA;
}


// Inversion achieved by subdividing the matrix A into blocks for faster LU computation
double* lu_blocks(double* A, unsigned int n) {
    double* iA = new double[n*n];
    double* PA = new double[n*n];
    double* P = pivoting_p(A,PA,n);
    
	// LU factorization by blocks
    block_factorization(PA, n);

	// Each loop works on a different column: can be paralleized
    #pragma omp parallel for schedule(dynamic)
    for (unsigned int j = 0; j < n; j++) {
        double sum;
        unsigned int i, l;
        double* x = new double[n]();
        double* y = new double[n]();

        // Lower triangular solve
        y[0] = P[j];
        for (i = 1; i < n; i++) {
            sum = 0;
            for (l = 0; l < i; l++)
                sum += PA[i*n+l] * y[l];
            y[i] = P[i*n+j] - sum;
        }

        // Upper triangular solve
        x[n-1] = y[n-1] / PA[n*n-1];
        for (i = n-2; i < n; i--) {
            sum = 0;
            for (l = i+1; l < n; l++)
                sum += PA[i*n+l] * x[l];
            x[i] = (y[i] - sum) / PA[i*n+i];
        }
        
        for (i = 0; i < n; i++)
            iA[i*n+j] = x[i];

        delete[] x;
        delete[] y;
    }

    delete[] P;
    delete[] PA;
	return iA;
}

// LU is performed by blocks of size BS (see documentation for implementation details)
void block_factorization(double* PA, unsigned int n) {
    unsigned int b, BS = 70;

    for (b = 0; b < n-BS; b += BS) {
        unsigned int size = b + BS;
        double* TMP = new double[BS*(n-size)];

        // STEP A
        for (unsigned int i = b; i < size; i++) {
            unsigned long in = i*n;
            for (unsigned int j = i+1; j < size; j++) {
                unsigned long jn = j*n;
                PA[jn+i] /= PA[in+i];
                for (unsigned int k = i+1; k < size; k++) {
                    PA[jn+k] -= PA[jn+i] * PA[in+k];
                }
            }
        }
        
        #pragma omp parallel
        {
            // STEP B
            #pragma omp single
            #pragma omp taskgroup
            {
                #pragma omp task
                {
                    // Solve lower A_01 = L_00 x U_01
                    #pragma omp taskloop
                    for (unsigned int j = size; j < n; j++) {  // for each column of A_01
                        double* x = new double[BS]();
                        x[0] = PA[b*n+j];
                        for (unsigned int i = 1; i < BS; i++) {
                            double sum = 0;
                            for (unsigned int l = 0; l < i; l++)
                                sum += PA[(b+i)*n+(b+l)] * x[l];  // L00
                            x[i] = PA[(b+i)*n+j] - sum;
                        }

                        for (unsigned int i = 1; i < BS; i++)
                            PA[(b+i)*n+j] = x[i];
                    }
                }

                #pragma omp task
                {
                    // Solve upper A_10 = L_10 x U_00 (using transpose looks like lower)
                    #pragma omp taskloop
                    for (unsigned int j = size; j < n; j++) {
                        unsigned long jn = j*n;
                        double* x = new double[BS]();

                        x[0] = PA[jn+b] / PA[b*(n+1)];
                        for (unsigned int i = 1; i < BS; i++) {
                            double sum = 0;
                            for (unsigned int l = 0; l < i; l++)
                                sum += PA[(b+l)*n+(b+i)] * x[l];  // U00
                            x[i] = (PA[jn+(b+i)] - sum) / PA[(b+i)*(n+1)];
                        }

                        for (unsigned int i = 0; i < BS; i++)
                            PA[jn+(b+i)] = x[i];
                    }
                }
            }

            // STEP C
            #pragma omp for
            for (unsigned int i = 0; i < n-size; i++) {
                unsigned long ii = i*BS;
                unsigned long is = i+size;
                for (unsigned int j = 0; j < BS; j++)
                    TMP[ii+j] = PA[(b+j)*n+is];
            }
            
            #pragma omp for collapse(2)
            for (unsigned int i = size; i < n; i++) {
                for (unsigned int j = size; j < n; j++) {
                    double sum = 0;
                    unsigned long in = i*n;
                    unsigned long jj = (j-size)*BS;
                    for (unsigned int k = b; k < size; k++)
                        sum += PA[in+k] * TMP[jj+(k-b)];
                    PA[in+j] -= sum;
                }
            }

            #pragma omp master
            delete[] TMP;
        }
    }
    
    // Final factorization
    for (unsigned int i = b; i < n; i++) {
        unsigned long in = i*n;
        for (unsigned int j = i+1; j < n; j++) {
            unsigned long jn = j*n;
            PA[jn+i] /= PA[in+i];
            for (unsigned int k = i+1; k < n; k++) {
                PA[jn+k] -= PA[jn+i] * PA[in+k];
            }
        }
    }
}
