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

double* prod_s(double* A, double* B, unsigned int rA, unsigned int cA, unsigned int cB) {
	double* C = new double[rA*cB];
	for (unsigned int i = 0; i < rA; i++)
		for (unsigned int j = 0; j < cB; j++)
			for (unsigned int k = 0; k < cA; k++)
				C[i*cB + j] += A[i*cA + k] * B[k*cB + j];
	return C;
}

double* prod_opt(double* A, double* B, unsigned int rA, unsigned int cA, unsigned int cB) {
	double* C = new double[rA*cB];
	double* tB = new double[cB*cA];
	unsigned long ii, jj;
	unsigned int i, j, k;
	double sum;

	for (i = 0; i < cB; i++)
		for (j = 0; j < cA; j++)
			tB[i*cA + j] = B[j*cB + i];
	
	for (i = 0; i < rA; i++) {
		for (j = 0; j < cB; j++) {
			ii = i*cA;
			jj = j*cA;

			sum = 0;
			for (k = 0; k < cA; k++)
				sum += A[ii + k] * tB[jj + k];
			C[i*cB + j] = sum;
		}
	}

	delete[] tB;
	return C;
}

double* prod_p(double* A, double* B, unsigned int rA, unsigned int cA, unsigned int cB) {
	double* C = new double[rA*cB];
	double* tB = new double[cB*cA];
	unsigned long ii, jj;
	unsigned int i, j, k;

	#pragma omp parallel
	{
		#pragma omp for collapse(2) private(i,j)
		for (i = 0; i < cB; i++)
			for (j = 0; j < cA; j++)
				tB[i*cA + j] = B[j*cB + i];
		
		#pragma omp for collapse(2) private(i,j,k,ii,jj) schedule(dynamic)
		for (i = 0; i < rA; i++) {
			for (j = 0; j < cB; j++) {
				ii = i*cA;
				jj = j*cA;

				double sum = 0;
				for (k = 0; k < cA; k++)
					sum += A[ii + k] * tB[jj + k];
				C[i*cB + j] = sum;
			}
		}
	}

	delete[] tB;
	return C;
}

int main(int argc, char** argv) {
	double t;
	double *A, *R;
	unsigned int n;

	if (argc < 2) {
		printf("Arguments:\n 1 - Matrix size (n)\n 2 - Number of threads (optional)\n");
		return 0;
	} else if (argc == 2) {
		if (atoi(argv[1]) <= 0)	return 1;
		n = atoi(argv[1]);
		omp_set_num_threads(omp_get_max_threads());
	} else if (argc == 3) {
		if (atoi(argv[1]) <= 0 || atoi(argv[2]) <= 0) return 2;
		n = atoi(argv[1]);
		omp_set_num_threads(atoi(argv[2]));
	}

	printf("[MATRIX MULTIPLICATION]\n");
    if (argc == 2)  printf("Threads: %d\n\n", omp_get_max_threads());
	A = get_random_matrix(n,n);

    t = omp_get_wtime();
    R = prod_s(A,A,n,n,n);
	printf("Serial:\t\t%lf\n", omp_get_wtime() - t);
	delete[] R;
	
    t = omp_get_wtime();
    R = prod_opt(A,A,n,n,n);
	printf("Optimized:\t%lf\n", omp_get_wtime() - t);
	delete[] R;

    t = omp_get_wtime();
    R = prod_p(A,A,n,n,n);
	printf("Parallel:\t%lf\n", omp_get_wtime() - t);
	delete[] R;

    return 0;
}