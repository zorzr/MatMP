#include <iostream>
#include <fstream>
#include <random>
#include <string>
#include <ctime>
#include <omp.h>
using namespace std;

double* get_random_matrix(unsigned int r, unsigned int c) {
    srand(time(NULL));
	double* A = new double[r*c];
	for (unsigned int i = 0; i < r*c; i++)
		A[i] = ((4.0 * (double) rand()) / (double) RAND_MAX) - 2;
    return A;
}

double* pivoting(double* A, double* PA, unsigned int n) {
    double* P = new double[n*n];
    unsigned int i, j, k;
    double t = omp_get_wtime();

    for (i = 0; i < n; i++) {
		for (j = 0; j < n; j++) {
			if (i == j) P[i*(n+1)] = 1;
			PA[i*n+j] = A[i*n+j];
        }
	}

    printf("Initialize:\t%lf\n",  omp_get_wtime()-t);
    t =  omp_get_wtime();

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
    
    printf("Pivoting:\t%lf\n",  omp_get_wtime()-t);
    return P;
}

double* lu_serial(double* A, unsigned int n) {
    double* PA = new double[n*n];
	unsigned int i, j, k, l;
    unsigned long in, jn;
    double t;

    // LU Factorization
    double* P = pivoting(A,PA,n);

    t = omp_get_wtime();
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
    printf("LU Fact.:\t%lf\n", omp_get_wtime()-t);
    t = omp_get_wtime();
    
	// Inversion
    double* iA = new double[n*n];
    double sum;

	for (j = 0; j < n; j++) {
		double* x = new double[n]();
		double* y = new double[n]();

		y[0] = P[j];
		for (i = 1; i < n; i++) {
			sum = 0;
			for (l = 0; l < i; l++)
				sum += PA[i*n+l] * y[l];
			y[i] = P[i*n+j] - sum;
		}

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
    printf("Inversion:\t%lf\n", omp_get_wtime()-t);

    delete[] P;
    delete[] PA;
	return iA;
}


// PARALLEL IMPLEMENTATIONS
double* pivoting_p(double* A, double* PA, unsigned int n) {
    double* P = new double[n*n];
    unsigned int i, j, k;
    double t = omp_get_wtime();

    #pragma omp parallel for collapse(2)
    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            if (i == j)  P[i*(n+1)] = 1;
            PA[i*n+j] = A[i*n+j];
        }
    }

    printf("Initialize:\t%lf\n",  omp_get_wtime()-t);
    t =  omp_get_wtime();

   
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

    printf("Pivoting:\t%lf\n",  omp_get_wtime()-t);
    return P;
}

double* lu_naive(double* A, unsigned int n) {
    double* iA = new double[n*n];
    double* PA = new double[n*n];
    double t;

    double* P = pivoting_p(A,PA,n);

    #pragma omp parallel
	{
        t = omp_get_wtime();
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
        #pragma omp master
        {
            printf("LU Fact.:\t%lf\n", omp_get_wtime()-t);
            t = omp_get_wtime();
        }

		// Inversion
		#pragma omp for schedule(dynamic)
		for (unsigned int j = 0; j < n; j++) {
			double sum;
			unsigned int i, l;
			double* x = new double[n]();
			double* y = new double[n]();

			y[0] = P[j];
			for (i = 1; i < n; i++) {
				sum = 0;
				for (l = 0; l < i; l++)
					sum += PA[i*n+l] * y[l];
				y[i] = P[i*n+j] - sum;
			}

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
    
    printf("Inversion:\t%lf\n", omp_get_wtime()-t);

    delete[] P;
    delete[] PA;
	return iA;
}

void block_factorization(double* PA, unsigned int n, unsigned int BS);
double* lu_blocks(double* A, unsigned int n, unsigned int BS) {
    double* iA = new double[n*n];
    double* PA = new double[n*n];

    double* P = pivoting_p(A,PA,n);

    double t = omp_get_wtime();
    block_factorization(PA, n, BS);
    printf("LU Fact.:\t%lf\n", omp_get_wtime()-t);
    t = omp_get_wtime();

    // Inversion
    #pragma omp parallel for schedule(dynamic)
    for (unsigned int j = 0; j < n; j++) {
        double sum;
        unsigned int i, l;
        double* x = new double[n]();
        double* y = new double[n]();

        y[0] = P[j];
        for (i = 1; i < n; i++) {
            sum = 0;
            for (l = 0; l < i; l++)
                sum += PA[i*n+l] * y[l];
            y[i] = P[i*n+j] - sum;
        }

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
    printf("Inversion:\t%lf\n", omp_get_wtime()-t);

    delete[] P;
    delete[] PA;
	return iA;
}

void block_factorization(double* PA, unsigned int n, unsigned int BS) {
    unsigned int b;

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
            #pragma omp single
            #pragma omp taskgroup
            {
                #pragma omp task
                {
                    // Solve lower A_01 = L_00 U_01
                    #pragma omp taskloop //num_tasks(omp_get_num_threads()/2)
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
                    // Solve upper (using transpose looks like lower)
                    #pragma omp taskloop //num_tasks(omp_get_num_threads()/2)
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




int main(int argc, char** argv) {
	double t;
	double *A, *R;
	unsigned int n, BS;

	if (argc < 3) {
		printf("Arguments:\n 1 - Matrix size (n)\n 2 - Block size (BS)\n 3 - Number of threads (optional)\n");
		return 0;
	} else if (argc == 3) {
		if (atoi(argv[1]) <= 0 || atoi(argv[2]) < 1) return 1;
		n = (unsigned int) atoi(argv[1]);
		BS = (unsigned int) atoi(argv[2]);
		omp_set_num_threads(omp_get_max_threads());
	} else if (argc == 4) {
		if (atoi(argv[1]) <= 0 || atoi(argv[2]) < 1 || atoi(argv[3]) <= 0) return 2;
		n = (unsigned int) atoi(argv[1]);
		BS = (unsigned int) atoi(argv[2]);
		omp_set_num_threads(atoi(argv[3]));
	}

	printf("[MATRIX INVERSION]\n");
    printf("Threads: %d\n\n", omp_get_max_threads());
	A = get_random_matrix(n,n);

    printf("[PARALLEL (NAIVE)]\n");
    t = omp_get_wtime();
    R = lu_naive(A,n);
	printf("OVERALL:\t%lf\n\n", omp_get_wtime() - t);
	delete[] R;

    printf("[PARALLEL (BLOCKS)]\n");
    t = omp_get_wtime();
    R = lu_blocks(A,n,BS);
	printf("OVERALL:\t%lf\n\n", omp_get_wtime() - t);
	delete[] R;

    printf("[SERIAL]\n");
    t = omp_get_wtime();
    R = lu_serial(A,n);
	printf("OVERALL:\t%lf\n", omp_get_wtime() - t);
	delete[] R;

    delete[] A;
    return 0;
}