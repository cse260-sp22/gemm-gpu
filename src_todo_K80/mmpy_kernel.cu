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
#define globB(x, y) A[x*N + y]

//__global__ void matMul(int N, _DOUBLE_ *C, _DOUBLE_ *A, _DOUBLE_ *B)
//{
//
//    int I = blockIdx.y * blockDim.y + threadIdx.y;
//    int J = blockIdx.x * blockDim.x + threadIdx.x;
//
//    if ((I < N) && (J < N))
//    {
//        _DOUBLE_ _c = 0;
//        for (unsigned int k = 0; k < N; k++)
//        {
//            _DOUBLE_ a = A[I * N + k];
//            _DOUBLE_ b = B[k * N + J];
//            _c += a * b;
//        }
//        C[I * N + J] = _c;
//    }
//}

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

	for (int kk = 0; kk < (N+TW-1)/TW; kk++){
	
		//Loading A
		if (I < N && (kk*TW + tx) < N){
			As[ty][tx] = A[I*N + kk*TW + tx];
		}
		else As[ty][tx] = 0;
		
		if ((I+16) < N && (kk*TW + tx) < N){
			As[ty+16][tx] = A[(I+16)*N + kk*TW + tx];
		}
		else As[ty+16][tx] = 0;
		
		if (I < N && (kk*TW + tx + 16) < N){
			As[ty][tx+16] = A[I*N + kk*TW + tx+16];
		}
		else As[ty][tx+16] = 0;

		if ((I+16) < N && (kk*TW + tx + 16) < N){
			As[ty+16][tx+16] = A[(I+16)*N + kk*TW + tx+16];
		}
		else As[ty+16][tx+16] = 0;

		//Loading B
		if ((kk*TW + ty) < N && J < N){
			Bs[ty][tx] = B[(kk*TW+ty)*N + J];
		}
		else Bs[ty][tx] = 0;

		if ((kk*TW + ty + 16) < N && J < N){
			Bs[ty+16][tx] = B[(kk*TW+ty+16)*N + J];
		}
		else Bs[ty+16][tx] = 0;

		if ((kk*TW + ty) < N && (J+16) < N){
			Bs[ty][tx+16] = B[(kk*TW+ty)*N + J+16];
		}
		else Bs[ty][tx+16] = 0;

		if ((kk*TW + ty + 16) < N && (J+16) < N){
			Bs[ty+16][tx+16] = B[(kk*TW+ty+16)*N + J+16];
		}
		else Bs[ty+16][tx+16] = 0;
		
		__syncthreads();

		for (int k = 0; k < TW; k++){
			Cij[0][0] += As[ty][k] * Bs[k][tx];
			Cij[1][0] += As[ty+16][k] * Bs[k][tx];
			Cij[0][1] += As[ty][k] * Bs[k][tx+16];
			Cij[1][1] += As[ty+16][k] * Bs[k][tx+16];
		}
		__syncthreads();
	}

	if (I < N && J < N){
		C[I*N + J] = Cij[0][0];
	}
	if ((I+16) < N && J < N){
		C[(I+16)*N + J] = Cij[1][0];
	}
	if (I < N && (J+16) < N){
		C[I*N + J+16] = Cij[0][1];
	}
	if ((I+16) < N && (J+16) < N){
		C[(I+16)*N + J+16] = Cij[1][1];
	}
}