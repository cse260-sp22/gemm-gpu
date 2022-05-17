// ;-*- mode: c;-*-
// Matrix multiply device code
#include <assert.h>
#include <math.h>
#include "../src/utils.h"
#include "../src/types.h"
#include "mytypes.h"

using namespace std;

#include <stdio.h>

#define Cy 32
#define Cx 32
#define Cc 16

#define ILP 2

#define globA(x, y) __ldg(&A[x*N + y])
#define globB(x, y) __ldg(&B[x*N + y])
#define globC(x, y) C[x*N + y]

__global__ void matMul(int N, _DOUBLE_ * __restrict C, _DOUBLE_ * __restrict A, _DOUBLE_ * __restrict B){

	//local shared storage
	__shared__ _DOUBLE_ As[Cy][Cc];
	__shared__ _DOUBLE_ Bs[Cc][Cx];

	const int tx = threadIdx.x;
	const int bx = ILP*blockIdx.x;

	const int ty = threadIdx.y;
	const int by = ILP*blockIdx.y;

	const int J = bx*blockDim.x + tx;
	const int I = by*blockDim.y + ty;

	_DOUBLE_ Cij[ILP][ILP] = {0};

	#pragma unroll
	for (int kk = 0; kk < (N+Cc-1)/Cc; kk++){
		#pragma unroll
		for (int load = 0; load < ILP; load ++){
				if (I + 16*ILP < N && kk*Cc + tx < N) As[ty + 16*ILP][tx] = globA((I + 16*ILP), (kk*Cc + tx)); else As[ty + 16*ILP][tx] = 0;

				if (kk*Cc + ty < N && J + 16*ILP < N) Bs[ty][tx + 16*ILP] = globB((kk*Cc + ty), (J + 16*ILP)); else Bs[ty][tx + 16*ILP] = 0;
		}	
		
		__syncthreads();

		for (int k = 0; k < Cc; k++){

			Cij[0] += As[ty]		[k] * Bs[k][tx];
			Cij[1] += As[ty + 16]	[k] * Bs[k][tx];
			Cij[2] += As[ty]		[k] * Bs[k][tx + 16];
			Cij[3] += As[ty + 16]	[k] * Bs[k][tx + 16];
        }
		__syncthreads();
	}

	if (I < N 		&& J < N) 		globC(I,        J)          = Cij[0];
	if (I + 16 < N 	&& J < N) 		globC((I + 16), J)          = Cij[1];
	if (I < N 		&& J + 16 < N) 	globC(I,        (J + 16))   = Cij[2];
	if (I + 16 < N 	&& J + 16 < N) 	globC((I + 16), (J + 16))   = Cij[3];

}
