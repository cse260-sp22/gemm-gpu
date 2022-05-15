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
#define globC(x, y) C[x*N + y]

__global__ void matMul(int N, _DOUBLE_ *C, _DOUBLE_ *A, _DOUBLE_ *B){

	//local shared storage
	__shared__ _DOUBLE_ As[TW][TW];
	__shared__ _DOUBLE_ Bs[TW][TW];

	int tx = threadIdx.x;
	int bx = blockIdx.x;

	int ty = threadIdx.y;
	int by = blockIdx.y;

	int J = 2*bx*blockDim.x + tx;
	int I = 2*by*blockDim.y + ty;

	_DOUBLE_ Cij[2][2] = {0};

	for (int kk = 0; kk < (N+TW-1)/TW; kk++){
	
		if (I       < N	&& kk*TW + tx       < N) As[ty]		[tx] 	    = globA(I, 			(kk*TW + tx)); 		else As[ty]		[tx] 		= 0;
		if (I + 16  < N && kk*TW + tx       < N) As[ty + 16][tx] 	    = globA((I + 16), 	(kk*TW + tx)); 		else As[ty + 16][tx] 		= 0;
		if (I       < N && kk*TW + tx + 16  < N) As[ty]		[tx + 16]   = globA(I, 			(kk*TW + tx + 16)); else As[ty]		[tx + 16] 	= 0;
		if (I + 16  < N && kk*TW + tx + 16  < N) As[ty + 16][tx + 16]   = globA((I + 16), 	(kk*TW + tx + 16)); else As[ty + 16][tx + 16] 	= 0;

		if (kk*TW + ty < N 		&& J < N) 		Bs[ty]		[tx] 		= globB((kk*TW+ty), 		J); 		else Bs[ty]		[tx] 		= 0;
		if (kk*TW + ty + 16 < N && J < N) 		Bs[ty + 16]	[tx] 		= globB((kk*TW+ty + 16), 	J); 		else Bs[ty + 16][tx] 		= 0;
		if (kk*TW + ty < N 		&& J + 16 < N) 	Bs[ty]		[tx + 16] 	= globB((kk*TW+ty), 		(J + 16)); 	else Bs[ty]		[tx + 16] 	= 0;
		if (kk*TW + ty + 16 < N && J + 16 < N) 	Bs[ty + 16]	[tx + 16] 	= globB((kk*TW+ty + 16), 	(J + 16)); 	else Bs[ty + 16][tx + 16] 	= 0;
		
		__syncthreads();

		for (int k = 0; k < TW; k++){
			Cij[0][0] += As[ty]		[k] * Bs[k][tx];
			Cij[1][0] += As[ty + 16][k] * Bs[k][tx];
			Cij[0][1] += As[ty]		[k] * Bs[k][tx + 16];
			Cij[1][1] += As[ty + 16][k] * Bs[k][tx + 16];
        }
		__syncthreads();
	}

	if (I < N 		&& J < N) 		globC(I,        J)          = Cij[0][0];
	if (I + 16 < N 	&& J < N) 		globC((I + 16), J)          = Cij[1][0];
	if (I < N 		&& J + 16 < N) 	globC(I,        (J + 16))   = Cij[0][1];
	if (I + 16 < N 	&& J + 16 < N) 	globC((I + 16), (J + 16))   = Cij[1][1];

}
