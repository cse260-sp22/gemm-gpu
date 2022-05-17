// ;-*- mode: c;-*-
// Matrix multiply device code
#include <assert.h>
#include <math.h>
#include "../src/utils.h"
#include "../src/types.h"
#include "mytypes.h"

using namespace std;

#include <stdio.h>

#define globA(x, y) __ldg(&A[x*N + y])
#define globB(x, y) __ldg(&B[x*N + y])
#define globC(x, y) C[x*N + y]

#define TW 32
#define Cy 32
#define Cx 32
#define Cc 16

__global__ void matMul_ilp(int N, _DOUBLE_ *C, _DOUBLE_ *A, _DOUBLE_ *B){

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

__global__ void matMul_old(int N, _DOUBLE_ *C, _DOUBLE_ *A, _DOUBLE_ *B){

	//local shared storage
	__shared__ double As[128][17];
	__shared__ double Bs[16][128];

	_DOUBLE_ Ar[8] 		= {0};
	_DOUBLE_ Br[8] 		= {0};
	_DOUBLE_ Cr[8][8] 	= {0};

	const int tx = threadIdx.x;
	const int bx = blockIdx.x*128;

	const int ty = threadIdx.y;
	const int by = blockIdx.y*128;

	const int thd_id = ty*16 + tx;

	const int warp_thd_id = thd_id % 32;
	const int warp_thd_id_x = 4*(warp_thd_id % 4);
	const int warp_thd_id_y = 4*(warp_thd_id / 4);

	const int warp_id = thd_id / 32;
	const int warp_id_x = 32*(warp_id % 4);
	const int warp_id_y = 64*(warp_id / 4);

	#pragma unroll
	for (int tl_id = 0; tl_id < N; tl_id += 16){
		
		#pragma unroll
		for (int num_ld = 0; num_ld < 128; num_ld += 16){
			if (by + ty + num_ld < N && tx + tl_id < N) As[ty + num_ld][tx] = globA((by + ty + num_ld), (tx + tl_id)); else As[ty + num_ld][tx] = 0;
			if (ty + tl_id < N && bx + tx + num_ld < N) Bs[ty][tx + num_ld] = globB((ty + tl_id), (bx + tx + num_ld)); else Bs[ty][tx + num_ld] = 0;
		}
		__syncthreads();

		#pragma unroll
		for (int prod = 0; prod < 16; prod++){
			#pragma unroll
			for (int ilp = 0; ilp < 4; ilp++){
				Ar[ilp] 		= As[warp_id_y + warp_thd_id_y + ilp]		[prod];
				Ar[ilp + 4] 	= As[warp_id_y + warp_thd_id_y + ilp + 32]	[prod];

				Br[ilp]		= Bs[prod][warp_id_x + warp_thd_id_x + ilp];
				Br[ilp + 4]	= Bs[prod][warp_id_x + warp_thd_id_x + ilp + 16];
			
			}

			#pragma unroll
			for (int ilpy = 0; ilpy < 8; ilpy++){
				#pragma unroll
				for (int ilpx = 0; ilpx < 8; ilpx++){
					Cr[ilpy][ilpx] += Ar[ilpy] * Br[ilpx];
				}
			}
			__syncthreads();
		}
	}

	#pragma unroll
	for (int str_y = 0; str_y < 4; str_y++){
		#pragma unroll
		for (int str_x = 0; str_x < 4; str_x++){
			if ((by + warp_id_y + warp_thd_id_y + str_y) < N && (bx + warp_id_x + warp_thd_id_x + str_x) < N)
				globC((by + warp_id_y + warp_thd_id_y + str_y), (bx + warp_id_x + warp_thd_id_x + str_x)) 			= Cr[str_y]		[str_x];	
			if ((by + warp_id_y + warp_thd_id_y + str_y) < N && (bx + warp_id_x + warp_thd_id_x + str_x + 16) < N)
				globC((by + warp_id_y + warp_thd_id_y + str_y), (bx + warp_id_x + warp_thd_id_x + str_x + 16)) 		= Cr[str_y]		[str_x + 4];	
			if ((by + warp_id_y + warp_thd_id_y + str_y + 32) < N && (bx + warp_id_x + warp_thd_id_x + str_x) < N)
				globC((by + warp_id_y + warp_thd_id_y + str_y + 32), (bx + warp_id_x + warp_thd_id_x + str_x)) 		= Cr[str_y + 4]	[str_x];	
			if ((by + warp_id_y + warp_thd_id_y + str_y + 32) < N && (bx + warp_id_x + warp_thd_id_x + str_x + 16) < N)
				globC((by + warp_id_y + warp_thd_id_y + str_y + 32), (bx + warp_id_x + warp_thd_id_x + str_x + 16)) = Cr[str_y + 4]	[str_x + 4];	
		}
	}
}

__global__ void matMul(const int N, _DOUBLE_ * __restrict C, _DOUBLE_ * __restrict A, _DOUBLE_ * __restrict B){

	//local shared storage
	__shared__ _DOUBLE_ As[64][17];
	__shared__ _DOUBLE_ Bs[16][64];

	_DOUBLE_ Ar[4] 		= {0};
	_DOUBLE_ Br[4] 		= {0};
	_DOUBLE_ Cr[4][4] 	= {0};

	const int tx = threadIdx.x;
	const int bx = blockIdx.x*64;

	const int ty = threadIdx.y;
	const int by = blockIdx.y*64;

	const int thd_id = ty*16 + tx;

	const int warp_thd_id = thd_id % 32;
	const int warp_thd_id_x = 2*(warp_thd_id % 4);
	const int warp_thd_id_y = 2*(warp_thd_id / 4);

	const int warp_id = thd_id / 32;
	const int warp_id_x = 16*(warp_id % 4);
	const int warp_id_y = 32*(warp_id / 4);

	#pragma unroll
	for (int tl_id = 0; tl_id < N; tl_id += 16){
		
		#pragma unroll
		for (int num_ld = 0; num_ld < 64; num_ld += 16){
			if (by + ty + num_ld < N && tx + tl_id < N) As[ty + num_ld][tx] = globA((by + ty + num_ld), (tx + tl_id)); else As[ty + num_ld][tx] = 0;
			if (ty + tl_id < N && bx + tx + num_ld < N) Bs[ty][tx + num_ld] = globB((ty + tl_id), (bx + tx + num_ld)); else Bs[ty][tx + num_ld] = 0;
		}
		__syncthreads();

		#pragma unroll
		for (int prod = 0; prod < 16; prod++){
			#pragma unroll
			for (int ilp = 0; ilp < 2; ilp++){
				Ar[ilp] 		= As[warp_id_y + warp_thd_id_y + ilp]		[prod];
				Ar[ilp + 2] 	= As[warp_id_y + warp_thd_id_y + ilp + 16]	[prod];

				Br[ilp]		= Bs[prod][warp_id_x + warp_thd_id_x + ilp];
				Br[ilp + 2]	= Bs[prod][warp_id_x + warp_thd_id_x + ilp + 8];
			
			}

			#pragma unroll
			for (int ilpy = 0; ilpy < 4; ilpy++){
				#pragma unroll
				for (int ilpx = 0; ilpx < 4; ilpx++){
					Cr[ilpy][ilpx] += Ar[ilpy] * Br[ilpx];
				}
			}
			__syncthreads();
		}
	}

	#pragma unroll
	for (int str_y = 0; str_y < 2; str_y++){
		#pragma unroll
		for (int str_x = 0; str_x < 2; str_x++){
			if ((by + warp_id_y + warp_thd_id_y + str_y) < N && (bx + warp_id_x + warp_thd_id_x + str_x) < N)
				globC((by + warp_id_y + warp_thd_id_y + str_y), (bx + warp_id_x + warp_thd_id_x + str_x)) 			= Cr[str_y]		[str_x];	

			if ((by + warp_id_y + warp_thd_id_y + str_y) < N && (bx + warp_id_x + warp_thd_id_x + str_x + 8) < N)
				globC((by + warp_id_y + warp_thd_id_y + str_y), (bx + warp_id_x + warp_thd_id_x + str_x + 8)) 		= Cr[str_y]		[str_x + 2];	

			if ((by + warp_id_y + warp_thd_id_y + str_y + 16) < N && (bx + warp_id_x + warp_thd_id_x + str_x) < N)
				globC((by + warp_id_y + warp_thd_id_y + str_y + 16), (bx + warp_id_x + warp_thd_id_x + str_x)) 		= Cr[str_y + 2]	[str_x];	

			if ((by + warp_id_y + warp_thd_id_y + str_y + 16) < N && (bx + warp_id_x + warp_thd_id_x + str_x + 8) < N)
				globC((by + warp_id_y + warp_thd_id_y + str_y + 16), (bx + warp_id_x + warp_thd_id_x + str_x + 8)) = Cr[str_y + 2]	[str_x + 2];	
		}
	}
}

// __global__ void matMul(int N, _DOUBLE_ *C, _DOUBLE_ *A, _DOUBLE_ *B) {
// 	if(N <= 512)
// 		matMul_ilp(N, C, A, B);
// 	else
// 		matMul_cutlass(N, C, A, B);
// }
