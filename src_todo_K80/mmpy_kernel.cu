// ;-*- mode: c;-*-
// Matrix multiply device code
#include <assert.h>
#include <math.h>
#include "../src/utils.h"
#include "../src/types.h"
#include "mytypes.h"

using namespace std;

#include <stdio.h>

#define TW 32

#define globA(x, y) A[x*N + y]
#define globB(x, y) B[x*N + y]

#define load_w_zero_padding(matrix, i, j, N)((i < N && j < N) ? matrix[(i * N) + j] : 0)

__global__ void matMul_naive(int N, _DOUBLE_ *C, _DOUBLE_ *A, _DOUBLE_ *B)
{

   int I = blockIdx.y * blockDim.y + threadIdx.y;
   int J = blockIdx.x * blockDim.x + threadIdx.x;

   if ((I < N) && (J < N))
   {
       _DOUBLE_ _c = 0;
       for (unsigned int k = 0; k < N; k++)
       {
           _DOUBLE_ a = A[I * N + k];
           _DOUBLE_ b = B[k * N + J];
           _c += a * b;
       }
       C[I * N + J] = _c;
   }
}

__global__ void matMul(int N, _DOUBLE_ *C, _DOUBLE_ *A, _DOUBLE_ *B){

	//local shared storage
	__shared__ double As[TW][TW];
	__shared__ double Bs[TW][TW];

	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int bx = blockIdx.x;
	int by = blockIdx.y;

	int J = bx*TW + tx;
	int I = by*TW + ty;

	double Cij[2][2] = {0};

	for (int kk = 0; kk < (N+TW-1)/TW; kk++) {
		// Here the i, j could be switched between A and B
		//Loading A
		#pragma unroll
		for(int i = 0; i < 2; i++) {
			#pragma unroll
			for(int j = 0; j < 2; j++) {
				As[ty + i * ILP_OFFSET][tx + j * ILP_OFFSET] = load_w_zero_padding(A, I + (i * ILP_OFFSET), (kk*TW + tx + (j * ILP_OFFSET)), N);
				// As[ty+ILP_OFFSET + i][tx + j] = load_w_zero_padding(A, (I + ILP_OFFSET + i), (kk*TW + tx + j), N);
				// As[ty + i][tx+ILP_OFFSET + j] = load_w_zero_padding(A, (I + i), (kk*TW + tx + ILP_OFFSET + j), N);
				// As[ty+ILP_OFFSET + i][tx+ILP_OFFSET + j] = load_w_zero_padding(A, (I + ILP_OFFSET + i), (kk*TW + tx + ILP_OFFSET + j), N);
			}
		}

		//Loading B
		#pragma unroll
		for(int i = 0; i < 2; i++) {
			#pragma unroll
			for(int j = 0; j < 2; j++) {
				Bs[ty + i * ILP_OFFSET][tx + j * ILP_OFFSET] = load_w_zero_padding(B, (kk*TW+ty + (i * ILP_OFFSET)), J + (j * ILP_OFFSET), N);
				// Bs[ty + ILP_OFFSET + i][tx + j] = load_w_zero_padding(B, (kk*TW+ty + ILP_OFFSET + i), J + j, N);
				// Bs[ty + i][tx + ILP_OFFSET + j] = load_w_zero_padding(B, (kk*TW+ty + i), J + ILP_OFFSET + j, N);
				// Bs[ty + ILP_OFFSET + i][tx + ILP_OFFSET + j] = load_w_zero_padding(B, (kk*TW+ty + ILP_OFFSET + i), J + ILP_OFFSET + j, N);
			}
		}

		__syncthreads();

		for (int k = 0; k < TW; k++){
			// Here the +1 could be shifted to A and vice-versa
			#pragma unroll
			for(int i = 0; i < 2; i++) {
				#pragma unroll
				for(int j = 0; j < 2; j++) {
					Cij[i][j] += As[ty + i * ILP_OFFSET][k] * Bs[k][tx + j * ILP_OFFSET];
				}
			}
			// The for loop above expands to:

			// Cij[0][0] += As[ty][k] * Bs[k][tx];
			// Cij[0][1] += As[ty][k] * Bs[k][tx + 1];
			// Cij[1][0] += As[ty + 1][k] * Bs[k][tx];
			// Cij[1][1] += As[ty + 1][k] * Bs[k][tx + 1];

			// Cij[0][2] += As[ty][k] * Bs[k][tx+ILP_OFFSET];
			// Cij[0][3] += As[ty][k] * Bs[k][tx+ILP_OFFSET + 1];
			// Cij[1][2] += As[ty + 1][k] * Bs[k][tx+ILP_OFFSET];
			// Cij[1][3] += As[ty + 1][k] * Bs[k][tx+ILP_OFFSET + 1];

			// Cij[2][0] += As[ty+ILP_OFFSET][k] * Bs[k][tx];
			// Cij[2][1] += As[ty+ILP_OFFSET][k] * Bs[k][tx + 1];
			// Cij[3][0] += As[ty+ILP_OFFSET + 1][k] * Bs[k][tx];
			// Cij[3][1] += As[ty+ILP_OFFSET + 1][k] * Bs[k][tx + 1];

			// Cij[2][2] += As[ty+ILP_OFFSET][k] * Bs[k][tx+ILP_OFFSET];
			// Cij[2][3] += As[ty+ILP_OFFSET][k] * Bs[k][tx+ILP_OFFSET + 1];
			// Cij[3][2] += As[ty+ILP_OFFSET + 1][k] * Bs[k][tx+ILP_OFFSET];
			// Cij[3][2] += As[ty+ILP_OFFSET + 1][k] * Bs[k][tx+ILP_OFFSET + 1];
		}
		__syncthreads();
	}

	if (I < N && J < N){
		C[I*N + J] = Cij[0][0];
	}
	if ((I+ILP_OFFSET) < N && J < N){
		C[(I+ILP_OFFSET)*N + J] = Cij[1][0];
	}
	if (I < N && (J+ILP_OFFSET) < N){
		C[I*N + J+ILP_OFFSET] = Cij[0][1];
	}
	if ((I+ILP_OFFSET) < N && (J+ILP_OFFSET) < N){
		C[(I+ILP_OFFSET)*N + J+ILP_OFFSET] = Cij[1][1];
	}
}
