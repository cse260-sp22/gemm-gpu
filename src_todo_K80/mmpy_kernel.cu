// ;-*- mode: c;-*-
// Matrix multiply device code
#include <assert.h>
#include <math.h>
#include "../src/utils.h"
#include "../src/types.h"
#include "mytypes.h"

using namespace std;

#include <stdio.h>

#define globA(x, y) __ldg(A[x*N + y])
#define globB(x, y) __ldg(B[x*N + y])
#define globC(x, y) __ldg(C[x*N + y])

__global__ void matMul(int N, _DOUBLE_ *C, _DOUBLE_ *A, _DOUBLE_ *B){

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
			As[ty + num_ld][tx] = globA((by + ty + num_ld), (tx + tl_id));
			Bs[ty][tx + i] = globB((ty + tl_id), (bx + tx + num_ld));
		}
		__syncthreads();

		#pragma unroll
		for (int prod = 0; prod < 16; prod++){
			#pragma unroll
			for (int ilp = 0; ilp < 4; ilp++){
				Ar[i] 		= As[warp_id_y + warp_thd_id_y + ilp]		[prod];
				Ar[i + 4] 	= As[warp_id_y + warp_thd_id_y + ilp + 32]	[prod];

				Br[i]		= Bs[prod][warp_id_x + warp_thd_id_x + ilp];
				Br[i + 4]	= Bs[prod][warp_id_x + warp_thd_id_x + ilp + 16];
			
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


