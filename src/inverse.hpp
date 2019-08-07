#ifndef INVERSE_HPP
#define INVERSE_HPP

// SERIAL IMPLEMENTATION
double* pivoting(double* A, double* PA, unsigned int n);
double* lu_serial(double* A, unsigned int n);


// PARALLEL IMPLEMENTATIONS
double* pivoting_p(double* A, double* PA, unsigned int n);
void block_factorization(double* PA, unsigned int n);

double* lu_naive(double* A, unsigned int n);
double* lu_blocks(double* A, unsigned int n);

#endif