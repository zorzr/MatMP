//-------------------------------------- SERIAL IMPLEMENTATION ---------------------------------------------

// Standard matrix multiplication: each element of C is given by the scalar 
//  product between a row of A and a column of B (three nested loops as follows).
double* prod_serial(double* A, double* B, unsigned int rA, unsigned int cA, unsigned int cB) {
	double* C = new double[rA*cB];
	for (unsigned int i = 0; i < rA; i++)
		for (unsigned int j = 0; j < cB; j++)
			for (unsigned int k = 0; k < cA; k++)
				C[i*cB + j] += A[i*cA + k] * B[k*cB + j];
	return C;
}

// Serial optimization: uses the transpose of B to reduce misses
double* prod_optimized(double* A, double* B, unsigned int rA, unsigned int cA, unsigned int cB) {
	double* C = new double[rA*cB];
	double* tB = new double[cB*cA];
	unsigned long ii, jj;
	unsigned int i, j, k;
	double sum;

	// Transposition of B
	for (i = 0; i < cB; i++)
		for (j = 0; j < cA; j++)
			tB[i*cA + j] = B[j*cB + i];

	// Scalar product between rows of A and B
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


//-----------------------------------------------PARALLEL IMPLEMENTATION--------------------------------------------

// For parallelization of the optimized serial code
double* prod_parallel(double* A, double* B, unsigned int rA, unsigned int cA, unsigned int cB) {
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
