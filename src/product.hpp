#ifndef PRODUCT_HPP
#define PRODUCT_HPP

double* prod_serial(double* A, double* B, unsigned int rA, unsigned int cA, unsigned int cB);
double* prod_optimized(double* A, double* B, unsigned int rA, unsigned int cA, unsigned int cB);
double* prod_parallel(double* A, double* B, unsigned int rA, unsigned int cA, unsigned int cB);

#endif