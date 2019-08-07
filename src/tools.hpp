#ifndef TOOLS_HPP
#define TOOLS_HPP

#include <iostream>
#include <fstream>
#include <random>
#include <string>
#include <ctime>
using namespace std;

double* get_random_matrix(unsigned int r, unsigned int c) {
	double* A = new double[r*c];

    srand(time(NULL));
	for (unsigned int i = 0; i < r*c; i++)
		A[i] = ((4.0 * (double) rand()) / (double) RAND_MAX) - 2;
    
    return A;
}

void check_identity(double* A, unsigned int n) {
    bool correct = true;
    
	for (unsigned int i = 0; i < n; i++) {
		for (unsigned int j = 0; j < n; j++) {
            if (i == j) {
                if (A[i*n+j] - 1.0 > 0.0001) {
                    correct = false;
                    break;
                }
            } else {
                if (A[i*n+j] > 0.0001) {
                    correct = false;
                    break;
                }
            }
        }
    }

    if (correct)  cout << "Success!" << endl;
    else  cout << "Method failed!" << endl;
}

#endif