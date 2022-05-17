// ;-*- mode: c;-*-
// Matrix multiply device code
#include <assert.h>
#include <math.h>
#include "../src/utils.h"
#include "../src/types.h"
#include "mytypes.h"

using namespace std;

#include <stdio.h>

#define Cy 64
#define Cx 64
#define Cc 64

#define ILP 8

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
				if (I + 64*load < N && kk*Cc + tx < N) As[ty + 64*load][tx] = globA((I + 64*load), (kk*Cc + tx)); else As[ty + 64*load][tx] = 0;
				if (I + 64*load < N && kk*Cc + tx + 64 < N) As[ty + 64*load][tx + 64] = globA((I + 64*load), (kk*Cc + tx + 64)); else As[ty + 64*load][tx + 64] = 0;

				if (kk*Cc + ty < N && J + 64*load < N) Bs[ty][tx + 64*load] = globB((kk*Cc + ty), (J + 64*load)); else Bs[ty][tx + 64*load] = 0;
				if (kk*Cc + ty + 64 < N && J + 64*load < N) Bs[ty + 64][tx + 64*load] = globB((kk*Cc + ty + 64), (J + 64*load)); else Bs[ty + 64][tx + 64*load] = 0;
		}	
		
		__syncthreads();
        #pragma unroll
		for (int k = 0; k < Cc; k++){
            #pragma unroll
            for (int i = 0; i < ILP; i++){
                #pragma unroll
                for (int j = 0; j < ILP; j++){
                    Cij[i][j] += As[ty + 32*j][k]*Bs[k][tx + 32*i]; 
                }
            }
        }
		__syncthreads();
	}

    #pragma unroll
    for (int i = 0; i < ILP; i++){
        #pragma unroll
        for (int j = 0; j < ILP; j++){
            if (I + 64*j < N && J + 64*i < N)
                globC((I + 64*j), (J + 64*i)) =  Cij[i][j]; 
        }
    }


}
