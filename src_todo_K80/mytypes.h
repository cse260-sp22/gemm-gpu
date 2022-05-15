#ifndef MYTYPES_H
#define MYTYPES_H

#define ILP_OFFSET 16

#define BLOCKTILE_M BLOCKDIM_Y/2
#define BLOCKTILE_N BLOCKDIM_X/2

#define MTILE 32
#define NTILE 32
#define KTILE 16

#define BLOCK_M 96
#define BLOCK_N 64
#define BLOCK_K 32

#define SUB_BLOCK_X (BLOCK_N / BLOCKDIM_X)
#define SUB_BLOCK_Y (BLOCK_M / BLOCKDIM_Y)

#endif
