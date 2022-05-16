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
#define Cy 32
#define Cx 32
#define Cc 32


#define globA(x, y) __ldg(&A[x*N + y])
#define globB(x, y) __ldg(&B[x*N + y])
#define globC(x, y) C[x*N + y]

__global__ void matMul(int N, _DOUBLE_ *C, _DOUBLE_ *A, _DOUBLE_ *B){

	//local shared storage
	__shared__ _DOUBLE_ As[Cy][Cc];
	__shared__ _DOUBLE_ Bs[Cc][Cx];

	int tx = threadIdx.x;
	int bx = blockIdx.x;

	int ty = threadIdx.y;
	int by = blockIdx.y;

	int J = 2*bx*blockDim.x + tx;
	int I = 2*by*blockDim.y + ty;

	_DOUBLE_ Cij[4] = {0};

	for (int kk = 0; kk < (N+Cc-1)/Cc; kk++){
	
		if (I       < N	&& kk*Cc + tx       < N) As[ty]		[tx] 	    = globA(I, 			(kk*Cc + tx)); 		else As[ty]		[tx] 		= 0;
		if (I + 16  < N && kk*Cc + tx       < N) As[ty + 16][tx] 	    = globA((I + 16), 	(kk*Cc + tx)); 		else As[ty + 16][tx] 		= 0;
		if (I       < N && kk*Cc + tx + 16  < N) As[ty]		[tx + 16]   = globA(I, 			(kk*Cc + tx + 16)); else As[ty]		[tx + 16] 	= 0;
		if (I + 16  < N && kk*Cc + tx + 16  < N) As[ty + 16][tx + 16]   = globA((I + 16), 	(kk*Cc + tx + 16)); else As[ty + 16][tx + 16] 	= 0;

		if (kk*Cc + ty < N 		&& J < N) 		Bs[ty]		[tx] 		= globB((kk*Cc+ty), 		J); 		else Bs[ty]		[tx] 		= 0;
		if (kk*Cc + ty + 16 < N && J < N) 		Bs[ty + 16]	[tx] 		= globB((kk*Cc+ty + 16), 	J); 		else Bs[ty + 16][tx] 		= 0;
		if (kk*Cc + ty < N 		&& J + 16 < N) 	Bs[ty]		[tx + 16] 	= globB((kk*Cc+ty), 		(J + 16)); 	else Bs[ty]		[tx + 16] 	= 0;
		if (kk*Cc + ty + 16 < N && J + 16 < N) 	Bs[ty + 16]	[tx + 16] 	= globB((kk*Cc+ty + 16), 	(J + 16)); 	else Bs[ty + 16][tx + 16] 	= 0;
		
		__syncthreads();

		for (int k = 0; k < TW; k++){

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
