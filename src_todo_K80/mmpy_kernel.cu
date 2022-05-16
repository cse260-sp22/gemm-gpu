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

#define load_w_zero_padding(matrix, i, j, N)((i) < N && (j) < N ? (matrix[(i)* N + (j)]) : 0)

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

__global__ void matMul_naive_naive(int N, _DOUBLE_ *C, _DOUBLE_ *A, _DOUBLE_ *B) {
	for(int i = 0; i < N; i++) {
		for(int j = 0; j < N; j++) {
			for(int k = 0; k < N; k++) {
				C[i*N + j] += A[i * N + k] * B[k * N + j];
			}
		}
	}
}

// __global__ void matMul(int N, _DOUBLE_ *C, _DOUBLE_ *A, _DOUBLE_ *B) {
// 	for(int mb = 0; mb < N; mb += MTILE) {
// 		for(int nb = 0; nb < N; nb += NTILE) {
// 			for(int kb = 0; kb < N; kb += KTILE) {
// 				// compute MTILE * NTILE * KTILE matrix product
// 				for(int k = 0; k < KTILE; k++) {
// 					for(int i = 0; i < MTILE; i++) {
// 						for(int j = 0; j < NTILE; j++) {
// 							int r = mb + i;
// 							int c = nb + j;

// 							C[r * N + c] += A[r * N + (kb + k)] * B[(kb + k) * N + c];
// 						}
// 					}
// 				}
// 			}
// 		}
// 	}
// }

__global__ void matMul(int N, _DOUBLE_ *C, _DOUBLE_ *A, _DOUBLE_ *B) {
	__shared__ _DOUBLE_ As[BLOCK_M][BLOCK_K];
	__shared__ _DOUBLE_ Bs[BLOCK_K][BLOCK_N];

	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int bx = blockIdx.x;
	int by = blockIdx.y;

	_DOUBLE_ Cij[SUB_BLOCK_Y][SUB_BLOCK_X] = {0};

	int I = by * BLOCK_M;
	int J = bx * BLOCK_N;

	for(int kk = 0; kk < N; kk += BLOCK_K) {
		// Load A into shared memory
		for(int i = 0; i < BLOCK_M; i += BLOCKDIM_Y) {
			for(int j = 0; j < BLOCK_K; j += BLOCKDIM_X) {
				As[ty + i][tx + j] = load_w_zero_padding(A, I + ty + i, kk + tx + j, N);
			}
		}

		// Load B into shared memory
		for(int i = 0; i < BLOCK_K; i += BLOCKDIM_Y) {
			for(int j = 0; j < BLOCK_N; j += BLOCKDIM_X) {
				Bs[ty + i][tx + j] = load_w_zero_padding(B, kk + ty + i, J + j + tx, N);
			}
		}
		__syncthreads();

		// Computing and accumulating block products
		for(int k = 0; k < BLOCK_K; k++) {
			for(int i = 0; i < SUB_BLOCK_Y; i++) {
				for(int j = 0; j < SUB_BLOCK_X; j++) {
					Cij[i][j] += As[i * BLOCKDIM_Y + ty][k] * Bs[k][j * BLOCKDIM_X + tx];
				}
			}
		}
		__syncthreads();
	}

	for(int i = 0; i < SUB_BLOCK_Y; i++) {
		for(int j = 0; j < SUB_BLOCK_X; j++) {
			int _i = (i * BLOCKDIM_Y) + I + ty;
			int _j = (j * BLOCKDIM_X) + J + tx;

			if(_i < N && _j < N) {
				C[(_i * N) + j] = Cij[i][j];
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