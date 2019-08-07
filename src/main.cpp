/*  MatMP - by zorzr and simmys
 *  Analysis and development of multithreading techniques for matrix multiplication and inversion.
 *  
 *  This file was made with the purpose of testing the correctness of the algorithms, rather than
 *  their actual speedup: finer details on the topic can be obtained from the "benchmark" folder.
 */

#include <omp.h>
#include "product.hpp"
#include "inverse.hpp"
#include "tools.hpp"

// PRODUCT
void prod_serial(double* A, unsigned int n) {
    cout << "[Serial]" << endl;
    double* iA = lu_naive(A,n);

    double t = omp_get_wtime();
    //double* I = prod_serial(A,iA,n,n,n);
	double* I = prod_optimized(A,iA,n,n,n);
	t = omp_get_wtime() - t;
    
    check_identity(I,n);
	printf("Elapsed time:  %lf\n", t);
	
    delete[] I;
	delete[] iA;
}

void prod_parallel(double* A, unsigned int n) {
    cout << "\n[Parallel]" << endl;
    double* iA = lu_naive(A,n);

    double t = omp_get_wtime();
    double* I = prod_parallel(A,iA,n,n,n);
	t = omp_get_wtime() - t;

    check_identity(I,n);
	printf("Elapsed time:  %lf\n", t);
	
    delete[] I;
	delete[] iA;
}

// INVERSE
void inv_serial(double* A, unsigned int n) {
    cout << "[Serial]" << endl;

    double t = omp_get_wtime();
    double* iA = lu_serial(A,n);
	t = omp_get_wtime() - t;

    double* I = prod_parallel(A,iA,n,n,n);
    check_identity(I,n);
	printf("Elapsed time:  %lf\n", t);
	
    delete[] I;
	delete[] iA;
}

void inv_parallel(double* A, unsigned int n) {
    cout << "\n[Parallel]" << endl;

    double t = omp_get_wtime();
    //double* iA = lu_naive(A,n);
    double* iA = lu_blocks(A,n);
	t = omp_get_wtime() - t;

    double* I = prod_parallel(A,iA,n,n,n);
    check_identity(I,n);
	printf("Elapsed time:  %lf\n", t);
	
    delete[] I;
	delete[] iA;
}

// TEST
void test_prod() {
	unsigned int n = 1000;
	double* A = get_random_matrix(n,n);
	
    cout << "[PRODUCT TEST]" << endl;
    prod_serial(A, n);
    prod_parallel(A, n);

	delete[] A;
}

void test_inverse() {
	unsigned int n = 1000;
	double* A = get_random_matrix(n,n);
	
    cout << "[INVERSE TEST]" << endl;
    inv_serial(A, n);
    inv_parallel(A, n);

	delete[] A;
}


int main(int argc, char** argv) {
	test_prod();
    cout << endl << endl;
    test_inverse();
	return 0;
}