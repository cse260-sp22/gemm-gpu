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

__global__ void matMul_ilp(int N, _DOUBLE_ *C, _DOUBLE_ *A, _DOUBLE_ *B){

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
		
		if ((I+ILP_OFFSET) < N && (kk*TW + tx) < N){
			As[ty+ILP_OFFSET][tx] = A[(I+ILP_OFFSET)*N + kk*TW + tx];
		}
		else As[ty+ILP_OFFSET][tx] = 0;
		
		if (I < N && (kk*TW + tx + ILP_OFFSET) < N){
			As[ty][tx+ILP_OFFSET] = A[I*N + kk*TW + tx+ILP_OFFSET];
		}
		else As[ty][tx+ILP_OFFSET] = 0;

		if ((I+ILP_OFFSET) < N && (kk*TW + tx + ILP_OFFSET) < N){
			As[ty+ILP_OFFSET][tx+ILP_OFFSET] = A[(I+ILP_OFFSET)*N + kk*TW + tx+ILP_OFFSET];
		}
		else As[ty+ILP_OFFSET][tx+ILP_OFFSET] = 0;

		//Loading B
		if ((kk*TW + ty) < N && J < N){
			Bs[ty][tx] = B[(kk*TW+ty)*N + J];
		}
		else Bs[ty][tx] = 0;

		if ((kk*TW + ty + ILP_OFFSET) < N && J < N){
			Bs[ty+ILP_OFFSET][tx] = B[(kk*TW+ty+ILP_OFFSET)*N + J];
		}
		else Bs[ty+ILP_OFFSET][tx] = 0;

		if ((kk*TW + ty) < N && (J+ILP_OFFSET) < N){
			Bs[ty][tx+ILP_OFFSET] = B[(kk*TW+ty)*N + J+ILP_OFFSET];
		}
		else Bs[ty][tx+ILP_OFFSET] = 0;

		if ((kk*TW + ty + ILP_OFFSET) < N && (J+ILP_OFFSET) < N){
			Bs[ty+ILP_OFFSET][tx+ILP_OFFSET] = B[(kk*TW+ty+ILP_OFFSET)*N + J+ILP_OFFSET];
		}
		else Bs[ty+ILP_OFFSET][tx+ILP_OFFSET] = 0;
		
		__syncthreads();

		for (int k = 0; k < TW; k++){
			Cij[0][0] += As[ty][k] * Bs[k][tx];
			Cij[1][0] += As[ty+ILP_OFFSET][k] * Bs[k][tx];
			Cij[0][1] += As[ty][k] * Bs[k][tx+ILP_OFFSET];
			Cij[1][1] += As[ty+ILP_OFFSET][k] * Bs[k][tx+ILP_OFFSET];
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

__global__ void matMul(int N, _DOUBLE_ *C, _DOUBLE_ *A, _DOUBLE_ *B) {
	for(int i = 0; i < N; i++) {
		for(int j = 0; j < N; j++) {
			for(int k = 0; k < N; k++) {
				C[i*N + j] += A[i * N + k] * B[k * N + j];
			}
		}
	}
}

// #define CUTLASS

// __global__ void matMul(int N, _DOUBLE_ *C, _DOUBLE_ *A, _DOUBLE_ *B) {
// 	#ifdef CUTLASS
// 		matMul_cutlass(N, C, A, B);
// 		#undef ILP
// 	#endif
// 	#ifdef ILP
// 		matMul_ilp(N, C, A, B);
// 	#endif
// }