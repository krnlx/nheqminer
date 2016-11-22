#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <functional>
#include <vector>
#include <iostream>
#include <stdint.h>
#include <mm_malloc.h>

#include "sa_cuda_context.hpp"
#include "param.h"
#include "sa_blake.h"



//#define THRD 64
#define BLK (NR_ROWS/THRD)


#define WN PARAM_N
#define WK PARAM_K

#define COLLISION_BIT_LENGTH (WN / (WK+1))
#define COLLISION_BYTE_LENGTH ((COLLISION_BIT_LENGTH+7)/8)
#define FINAL_FULL_WIDTH (2*COLLISION_BYTE_LENGTH+sizeof(uint32_t)*(1 << (WK)))

#define NDIGITS   (WK+1)
#define DIGITBITS (WN/(NDIGITS))
#define PROOFSIZE (1u<<WK)
#define COMPRESSED_PROOFSIZE ((COLLISION_BIT_LENGTH+1)*PROOFSIZE*4/(8*sizeof(uint32_t)))



//orig defines

#if NR_ROWS_LOG <= 16 && NR_SLOTS <= (1 << 8)

#define ENCODE_INPUTS(row, slot0, slot1) \
    ((row << 16) | ((slot1 & 0xff) << 8) | (slot0 & 0xff))
#define DECODE_ROW(REF)   (REF >> 16)
#define DECODE_SLOT1(REF) ((REF >> 8) & 0xff)
#define DECODE_SLOT0(REF) (REF & 0xff)

#elif NR_ROWS_LOG == 18 && NR_SLOTS <= (1 << 7)

#define ENCODE_INPUTS(row, slot0, slot1) \
    ((row << 14) | ((slot1 & 0x7f) << 7) | (slot0 & 0x7f))
#define DECODE_ROW(REF)   (REF >> 14)
#define DECODE_SLOT1(REF) ((REF >> 7) & 0x7f)
#define DECODE_SLOT0(REF) (REF & 0x7f)

#elif NR_ROWS_LOG == 19 && NR_SLOTS <= (1 << 6)

#define ENCODE_INPUTS(row, slot0, slot1) \
    ((row << 13) | ((slot1 & 0x3f) << 6) | (slot0 & 0x3f)) /* 1 spare bit */
#define DECODE_ROW(REF)   (REF >> 13)
#define DECODE_SLOT1(REF) ((REF >> 6) & 0x3f)
#define DECODE_SLOT0(REF) (REF & 0x3f)

#elif NR_ROWS_LOG == 20 && NR_SLOTS <= (1 << 6)

#define ENCODE_INPUTS(row, slot0, slot1) \
    ((row << 12) | ((slot1 & 0x3f) << 6) | (slot0 & 0x3f))
#define DECODE_ROW(REF)   (REF >> 12)
#define DECODE_SLOT1(REF) ((REF >> 6) & 0x3f)
#define DECODE_SLOT0(REF) (REF & 0x3f)

#else
#error "unsupported NR_ROWS_LOG"
#endif



#define nv64to32(low,high,X) asm volatile( "mov.b64 {%0,%1}, %2; \n\t" : "=r"(low), "=r"(high) : "l"(X))
#define nv32to64(X,low,high) asm volatile( "mov.b64 %0,{%1, %2}; \n\t": "=l"(X) : "r"(low), "r"(high))


__constant__ ulong blake_iv[] =
{
	0x6a09e667f3bcc908, 0xbb67ae8584caa73b,
	0x3c6ef372fe94f82b, 0xa54ff53a5f1d36f1,
	0x510e527fade682d1, 0x9b05688c2b3e6c1f,
	0x1f83d9abfb41bd6b, 0x5be0cd19137e2179,
};

//OPENCL TO CUDA

#define __global
#define __local __shared__
#define get_global_id(F) (blockIdx.x * blockDim.x + threadIdx.x)
#define get_global_size(F) (gridDim.x * blockDim.x)
#define get_local_id(F) (threadIdx.x)
#define get_local_size(F) (blockDim.x)
#define barrier(F) __syncthreads()

//#define barrier(F) __threadfence()

#define atomic_add(A,X) atomicAdd(A,X)
#define atomic_inc(A) atomic_add(A,1)
#define atomic_sub(A,X) atomicSub(A,X)
#define atomic_dec(A) atomic_sub(A,1)

__global__
void kernel_init_ht(__global char *ht, __global uint *rowCounters)
{
    rowCounters[get_global_id(0)] = 0;
}


__device__ __forceinline__
uint ht_store(uint round, __global char *ht, uint i,
	ulong xi0, ulong xi1, ulong xi2, ulong xi3, __global uint *rowCounters)
{
    uint    row;
    __global char       *p;
    uint                cnt;
#if NR_ROWS_LOG == 16
    if (!(round % 2))
	row = (xi0 & 0xffff);
    else
	// if we have in hex: "ab cd ef..." (little endian xi0) then this
	// formula computes the row as 0xdebc. it skips the 'a' nibble as it
	// is part of the PREFIX. The Xi will be stored starting with "ef...";
	// 'e' will be considered padding and 'f' is part of the current PREFIX
	row = ((xi0 & 0xf00) << 4) | ((xi0 & 0xf00000) >> 12) |
	    ((xi0 & 0xf) << 4) | ((xi0 & 0xf000) >> 12);
#elif NR_ROWS_LOG == 18
    if (!(round % 2))
	row = (xi0 & 0xffff) | ((xi0 & 0xc00000) >> 6);
    else
	row = ((xi0 & 0xc0000) >> 2) |
	    ((xi0 & 0xf00) << 4) | ((xi0 & 0xf00000) >> 12) |
	    ((xi0 & 0xf) << 4) | ((xi0 & 0xf000) >> 12);
#elif NR_ROWS_LOG == 19
    if (!(round % 2))
	row = (xi0 & 0xffff) | ((xi0 & 0xe00000) >> 5);
    else
	row = ((xi0 & 0xe0000) >> 1) |
	    ((xi0 & 0xf00) << 4) | ((xi0 & 0xf00000) >> 12) |
	    ((xi0 & 0xf) << 4) | ((xi0 & 0xf000) >> 12);
#elif NR_ROWS_LOG == 20
    if (!(round % 2))
	row = (xi0 & 0xffff) | ((xi0 & 0xf00000) >> 4);
    else
	row = ((xi0 & 0xf0000) >> 0) |
	    ((xi0 & 0xf00) << 4) | ((xi0 & 0xf00000) >> 12) |
	    ((xi0 & 0xf) << 4) | ((xi0 & 0xf000) >> 12);
#else
#error "unsupported NR_ROWS_LOG"
#endif
    xi0 = (xi0 >> 16) | (xi1 << (64 - 16));
    xi1 = (xi1 >> 16) | (xi2 << (64 - 16));
    xi2 = (xi2 >> 16) | (xi3 << (64 - 16));
    p = ht + row * NR_SLOTS * SLOT_LEN;
    uint rowIdx = row/ROWS_PER_UINT;
    uint rowOffset = BITS_PER_ROW*(row%ROWS_PER_UINT);
    uint xcnt = atomic_add(rowCounters + rowIdx, 1 << rowOffset);
    xcnt = (xcnt >> rowOffset) & ROW_MASK;
    cnt = xcnt;
    if (cnt >= NR_SLOTS)
      {
	// avoid overflows
	atomic_sub(rowCounters + rowIdx, 1 << rowOffset);
	return 1;
      }
    p += cnt * SLOT_LEN + xi_offset_for_round(round);
    // store "i" (always 4 bytes before Xi)
    *(__global uint *)(p - 4) = i;
    if (round == 0 || round == 1)
      {
	// store 24 bytes
	*(__global ulong *)(p + 0) = xi0;
	*(__global ulong *)(p + 8) = xi1;
	*(__global ulong *)(p + 16) = xi2;
      }
    else if (round == 2)
      {
	// store 20 bytes
	*(__global uint *)(p + 0) = xi0;
	*(__global ulong *)(p + 4) = (xi0 >> 32) | (xi1 << 32);
	*(__global ulong *)(p + 12) = (xi1 >> 32) | (xi2 << 32);
      }
    else if (round == 3)
      {
	// store 16 bytes
	*(__global uint *)(p + 0) = xi0;
	*(__global ulong *)(p + 4) = (xi0 >> 32) | (xi1 << 32);
	*(__global uint *)(p + 12) = (xi1 >> 32);
      }
    else if (round == 4)
      {
	// store 16 bytes
	*(__global ulong *)(p + 0) = xi0;
	*(__global ulong *)(p + 8) = xi1;
      }
    else if (round == 5)
      {
	// store 12 bytes
	*(__global ulong *)(p + 0) = xi0;
	*(__global uint *)(p + 8) = xi1;
      }
    else if (round == 6 || round == 7)
      {
	// store 8 bytes
	*(__global uint *)(p + 0) = xi0;
	*(__global uint *)(p + 4) = (xi0 >> 32);
      }
    else if (round == 8)
      {
	// store 4 bytes
	*(__global uint *)(p + 0) = xi0;
      }
    return 0;
}



#define rotate(a, bits) ((a) << (bits)) | ((a) >> (64 - (bits)))

#define mix(va, vb, vc, vd, x, y) \
    va = (va + vb + x); \
    vd = rotate((vd ^ va), 64 - 32); \
    vc = (vc + vd); \
    vb = rotate((vb ^ vc), 64 - 24); \
    va = (va + vb + y); \
    vd = rotate((vd ^ va), 64 - 16); \
    vc = (vc + vd); \
    vb = rotate((vb ^ vc), 64 - 63);

__global__ 
void kernel_round0(__global ulong *blake_state, __global char *ht,
	__global uint *rowCounters, __global uint *debug)
{
    uint                tid = get_global_id(0);
    ulong               v[16];
    uint                inputs_per_thread = NR_INPUTS / get_global_size(0);
    uint                input = tid * inputs_per_thread;
    uint                input_end = (tid + 1) * inputs_per_thread;
    uint                dropped = 0;
    while (input < input_end)
      {
	// shift "i" to occupy the high 32 bits of the second ulong word in the
	// message block
	ulong word1 = (ulong)input << 32;
	// init vector v
	v[0] = blake_state[0];
	v[1] = blake_state[1];
	v[2] = blake_state[2];
	v[3] = blake_state[3];
	v[4] = blake_state[4];
	v[5] = blake_state[5];
	v[6] = blake_state[6];
	v[7] = blake_state[7];
	v[8] =  blake_iv[0];
	v[9] =  blake_iv[1];
	v[10] = blake_iv[2];
	v[11] = blake_iv[3];
	v[12] = blake_iv[4];
	v[13] = blake_iv[5];
	v[14] = blake_iv[6];
	v[15] = blake_iv[7];
	// mix in length of data
	v[12] ^= ZCASH_BLOCK_HEADER_LEN + 4 /* length of "i" */;
	// last block
	v[14] ^= (ulong)-1;

	// round 1
	mix(v[0], v[4], v[8],  v[12], 0, word1);
	mix(v[1], v[5], v[9],  v[13], 0, 0);
	mix(v[2], v[6], v[10], v[14], 0, 0);
	mix(v[3], v[7], v[11], v[15], 0, 0);
	mix(v[0], v[5], v[10], v[15], 0, 0);
	mix(v[1], v[6], v[11], v[12], 0, 0);
	mix(v[2], v[7], v[8],  v[13], 0, 0);
	mix(v[3], v[4], v[9],  v[14], 0, 0);
	// round 2
	mix(v[0], v[4], v[8],  v[12], 0, 0);
	mix(v[1], v[5], v[9],  v[13], 0, 0);
	mix(v[2], v[6], v[10], v[14], 0, 0);
	mix(v[3], v[7], v[11], v[15], 0, 0);
	mix(v[0], v[5], v[10], v[15], word1, 0);
	mix(v[1], v[6], v[11], v[12], 0, 0);
	mix(v[2], v[7], v[8],  v[13], 0, 0);
	mix(v[3], v[4], v[9],  v[14], 0, 0);
	// round 3
	mix(v[0], v[4], v[8],  v[12], 0, 0);
	mix(v[1], v[5], v[9],  v[13], 0, 0);
	mix(v[2], v[6], v[10], v[14], 0, 0);
	mix(v[3], v[7], v[11], v[15], 0, 0);
	mix(v[0], v[5], v[10], v[15], 0, 0);
	mix(v[1], v[6], v[11], v[12], 0, 0);
	mix(v[2], v[7], v[8],  v[13], 0, word1);
	mix(v[3], v[4], v[9],  v[14], 0, 0);
	// round 4
	mix(v[0], v[4], v[8],  v[12], 0, 0);
	mix(v[1], v[5], v[9],  v[13], 0, word1);
	mix(v[2], v[6], v[10], v[14], 0, 0);
	mix(v[3], v[7], v[11], v[15], 0, 0);
	mix(v[0], v[5], v[10], v[15], 0, 0);
	mix(v[1], v[6], v[11], v[12], 0, 0);
	mix(v[2], v[7], v[8],  v[13], 0, 0);
	mix(v[3], v[4], v[9],  v[14], 0, 0);
	// round 5
	mix(v[0], v[4], v[8],  v[12], 0, 0);
	mix(v[1], v[5], v[9],  v[13], 0, 0);
	mix(v[2], v[6], v[10], v[14], 0, 0);
	mix(v[3], v[7], v[11], v[15], 0, 0);
	mix(v[0], v[5], v[10], v[15], 0, word1);
	mix(v[1], v[6], v[11], v[12], 0, 0);
	mix(v[2], v[7], v[8],  v[13], 0, 0);
	mix(v[3], v[4], v[9],  v[14], 0, 0);
	// round 6
	mix(v[0], v[4], v[8],  v[12], 0, 0);
	mix(v[1], v[5], v[9],  v[13], 0, 0);
	mix(v[2], v[6], v[10], v[14], 0, 0);
	mix(v[3], v[7], v[11], v[15], 0, 0);
	mix(v[0], v[5], v[10], v[15], 0, 0);
	mix(v[1], v[6], v[11], v[12], 0, 0);
	mix(v[2], v[7], v[8],  v[13], 0, 0);
	mix(v[3], v[4], v[9],  v[14], word1, 0);
	// round 7
	mix(v[0], v[4], v[8],  v[12], 0, 0);
	mix(v[1], v[5], v[9],  v[13], word1, 0);
	mix(v[2], v[6], v[10], v[14], 0, 0);
	mix(v[3], v[7], v[11], v[15], 0, 0);
	mix(v[0], v[5], v[10], v[15], 0, 0);
	mix(v[1], v[6], v[11], v[12], 0, 0);
	mix(v[2], v[7], v[8],  v[13], 0, 0);
	mix(v[3], v[4], v[9],  v[14], 0, 0);
	// round 8
	mix(v[0], v[4], v[8],  v[12], 0, 0);
	mix(v[1], v[5], v[9],  v[13], 0, 0);
	mix(v[2], v[6], v[10], v[14], 0, word1);
	mix(v[3], v[7], v[11], v[15], 0, 0);
	mix(v[0], v[5], v[10], v[15], 0, 0);
	mix(v[1], v[6], v[11], v[12], 0, 0);
	mix(v[2], v[7], v[8],  v[13], 0, 0);
	mix(v[3], v[4], v[9],  v[14], 0, 0);
	// round 9
	mix(v[0], v[4], v[8],  v[12], 0, 0);
	mix(v[1], v[5], v[9],  v[13], 0, 0);
	mix(v[2], v[6], v[10], v[14], 0, 0);
	mix(v[3], v[7], v[11], v[15], 0, 0);
	mix(v[0], v[5], v[10], v[15], 0, 0);
	mix(v[1], v[6], v[11], v[12], 0, 0);
	mix(v[2], v[7], v[8],  v[13], word1, 0);
	mix(v[3], v[4], v[9],  v[14], 0, 0);
	// round 10
	mix(v[0], v[4], v[8],  v[12], 0, 0);
	mix(v[1], v[5], v[9],  v[13], 0, 0);
	mix(v[2], v[6], v[10], v[14], 0, 0);
	mix(v[3], v[7], v[11], v[15], word1, 0);
	mix(v[0], v[5], v[10], v[15], 0, 0);
	mix(v[1], v[6], v[11], v[12], 0, 0);
	mix(v[2], v[7], v[8],  v[13], 0, 0);
	mix(v[3], v[4], v[9],  v[14], 0, 0);
	// round 11
	mix(v[0], v[4], v[8],  v[12], 0, word1);
	mix(v[1], v[5], v[9],  v[13], 0, 0);
	mix(v[2], v[6], v[10], v[14], 0, 0);
	mix(v[3], v[7], v[11], v[15], 0, 0);
	mix(v[0], v[5], v[10], v[15], 0, 0);
	mix(v[1], v[6], v[11], v[12], 0, 0);
	mix(v[2], v[7], v[8],  v[13], 0, 0);
	mix(v[3], v[4], v[9],  v[14], 0, 0);
	// round 12
	mix(v[0], v[4], v[8],  v[12], 0, 0);
	mix(v[1], v[5], v[9],  v[13], 0, 0);
	mix(v[2], v[6], v[10], v[14], 0, 0);
	mix(v[3], v[7], v[11], v[15], 0, 0);
	mix(v[0], v[5], v[10], v[15], word1, 0);
	mix(v[1], v[6], v[11], v[12], 0, 0);
	mix(v[2], v[7], v[8],  v[13], 0, 0);
	mix(v[3], v[4], v[9],  v[14], 0, 0);

	// compress v into the blake state; this produces the 50-byte hash
	// (two Xi values)
	ulong h[7];
	h[0] = blake_state[0] ^ v[0] ^ v[8];
	h[1] = blake_state[1] ^ v[1] ^ v[9];
	h[2] = blake_state[2] ^ v[2] ^ v[10];
	h[3] = blake_state[3] ^ v[3] ^ v[11];
	h[4] = blake_state[4] ^ v[4] ^ v[12];
	h[5] = blake_state[5] ^ v[5] ^ v[13];
	h[6] = (blake_state[6] ^ v[6] ^ v[14]) & 0xffff;

	// store the two Xi values in the hash table
#if ZCASH_HASH_LEN == 50
	dropped += ht_store(0, ht, input * 2,
		h[0],
		h[1],
		h[2],
		h[3], rowCounters);
	dropped += ht_store(0, ht, input * 2 + 1,
		(h[3] >> 8) | (h[4] << (64 - 8)),
		(h[4] >> 8) | (h[5] << (64 - 8)),
		(h[5] >> 8) | (h[6] << (64 - 8)),
		(h[6] >> 8), rowCounters);
#else
#error "unsupported ZCASH_HASH_LEN"
#endif

	input++;
      }
#ifdef ENABLE_DEBUG
    debug[tid * 2] = 0;
    debug[tid * 2 + 1] = dropped;
#endif
}


__device__ __forceinline__ ulong half_aligned_long(ulong *p, uint offset)
{
	return
		(((ulong)*(uint *)((char *)p + offset + 0)) << 0) |
		(((ulong)*(uint *)((char *)p + offset + 4)) << 32);
}

/*
** Access a well-aligned int.
*/
__device__ __forceinline__ uint well_aligned_int(ulong *_p, uint offset)
{
	char *p = (char *)_p;
	return *(uint *)(p + offset);
}

/*
** XOR a pair of Xi values computed at "round - 1" and store the result in the
** hash table being built for "round". Note that when building the table for
** even rounds we need to skip 1 padding byte present in the "round - 1" table
** (the "0xAB" byte mentioned in the description at the top of this file.) But
** also note we can't load data directly past this byte because this would
** cause an unaligned memory access which is undefined per the OpenCL spec.
**
** Return 0 if successfully stored, or 1 if the row overflowed.
*/
__device__ __forceinline__ uint get_row_nr_4(uint xi0, uint round){
uint row;
#if NR_ROWS_LOG == 16
    if (!(round % 2))
        row = (xi0 & 0xffff);
    else
        // if we have in hex: "ab cd ef..." (little endian xi0) then this
        // formula computes the row as 0xdebc. it skips the 'a' nibble as it
        // is part of the PREFIX. The Xi will be stored starting with "ef...";
        // 'e' will be considered padding and 'f' is part of the current PREFIX
        row = ((xi0 & 0xf00) << 4) | ((xi0 & 0xf00000) >> 12) |
            ((xi0 & 0xf) << 4) | ((xi0 & 0xf000) >> 12);
#elif NR_ROWS_LOG == 18
    if (!(round % 2))
        row = (xi0 & 0xffff) | ((xi0 & 0xc00000) >> 6);
    else
        row = ((xi0 & 0xc0000) >> 2) |
            ((xi0 & 0xf00) << 4) | ((xi0 & 0xf00000) >> 12) |
            ((xi0 & 0xf) << 4) | ((xi0 & 0xf000) >> 12);
#elif NR_ROWS_LOG == 19
    if (!(round % 2))
        row = (xi0 & 0xffff) | ((xi0 & 0xe00000) >> 5);
    else
        row = ((xi0 & 0xe0000) >> 1) |
            ((xi0 & 0xf00) << 4) | ((xi0 & 0xf00000) >> 12) |
            ((xi0 & 0xf) << 4) | ((xi0 & 0xf000) >> 12);
#elif NR_ROWS_LOG == 20
    if (!(round % 2))
        row = (xi0 & 0xffff) | ((xi0 & 0xf00000) >> 4);
    else
        row = ((xi0 & 0xf0000) >> 0) |
            ((xi0 & 0xf00) << 4) | ((xi0 & 0xf00000) >> 12) |
            ((xi0 & 0xf) << 4) | ((xi0 & 0xf000) >> 12);
#else
#error "unsupported NR_ROWS_LOG"
#endif
        return row;
}

__device__ __forceinline__
uint get_row_nr_8(ulong xi0, uint round){
uint row;
#if NR_ROWS_LOG == 16
    if (!(round % 2))
	row = (xi0 & 0xffff);
    else
	// if we have in hex: "ab cd ef..." (little endian xi0) then this
	// formula computes the row as 0xdebc. it skips the 'a' nibble as it
	// is part of the PREFIX. The Xi will be stored starting with "ef...";
	// 'e' will be considered padding and 'f' is part of the current PREFIX
	row = ((xi0 & 0xf00) << 4) | ((xi0 & 0xf00000) >> 12) |
	    ((xi0 & 0xf) << 4) | ((xi0 & 0xf000) >> 12);
#elif NR_ROWS_LOG == 18
    if (!(round % 2))
	row = (xi0 & 0xffff) | ((xi0 & 0xc00000) >> 6);
    else
	row = ((xi0 & 0xc0000) >> 2) |
	    ((xi0 & 0xf00) << 4) | ((xi0 & 0xf00000) >> 12) |
	    ((xi0 & 0xf) << 4) | ((xi0 & 0xf000) >> 12);
#elif NR_ROWS_LOG == 19
    if (!(round % 2))
	row = (xi0 & 0xffff) | ((xi0 & 0xe00000) >> 5);
    else
	row = ((xi0 & 0xe0000) >> 1) |
	    ((xi0 & 0xf00) << 4) | ((xi0 & 0xf00000) >> 12) |
	    ((xi0 & 0xf) << 4) | ((xi0 & 0xf000) >> 12);
#elif NR_ROWS_LOG == 20
    if (!(round % 2))
	row = (xi0 & 0xffff) | ((xi0 & 0xf00000) >> 4);
    else
	row = ((xi0 & 0xf0000) >> 0) |
	    ((xi0 & 0xf00) << 4) | ((xi0 & 0xf00000) >> 12) |
	    ((xi0 & 0xf) << 4) | ((xi0 & 0xf000) >> 12);
#else
#error "unsupported NR_ROWS_LOG"
#endif
	return row;
}

__device__ __forceinline__
void store8(__global char *p,ulong store){
	asm volatile ( "st.global.cs.b64  [%0], %1;\n\t" :: "l"(p), "l" (store));
}

__device__ __forceinline__
void store4(__global char *p,uint store){
        asm volatile ( "st.global.cs.b32  [%0], %1;\n\t" :: "l"(p), "r" (store));
}

__device__ __forceinline__
void store_ulong2(__global char *p,ulong2 store){
	asm volatile ( "st.global.cs.v2.b64  [%0],{ %1, %2 };\n\t" :: "l"(p), "l" (store.x), "l" (store.y));
}

__device__ __forceinline__
void store_uint2(__global char *p,uint2 store){
        asm volatile ( "st.global.cs.v2.b32  [%0],{ %1, %2 };\n\t" :: "l"(p), "r" (store.x), "r" (store.y));
}

__device__ __forceinline__
void store_uint4(__global char *p,uint4 store){
        asm volatile ( "st.global.cs.v4.b32  [%0],{ %1, %2, %3, %4 };\n\t" :: "l"(p), "r" (store.x), "r" (store.y), "r" (store.z), "r" (store.w));
}

__device__ __forceinline__
ulong load8_last(__global ulong *p,uint offset){
	p=(__global ulong *)((__global char *)p + offset); 
        ulong r;
        asm volatile ( "ld.global.cs.nc.b64  %0, [%1];\n\t" : "=l"(r) : "l"(p));
        return r;
}

__device__ __forceinline__
ulong load8(__global ulong *p,uint offset){
	p=(__global ulong *)((__global char *)p + offset); 
        ulong r;
        asm volatile ( "ld.global.cs.nc.b64  %0, [%1];\n\t" : "=l"(r) : "l"(p));
        return r;
}


__device__ __forceinline__
ulong2 load16l(__global ulong *p,uint offset){
        p=(__global ulong *)((__global char *)p + offset); 
        ulong2 r;
        asm volatile ( "ld.global.cs.nc.v2.b64  {%0,%1}, [%2];\n\t" : "=l"(r.x), "=l"(r.y) : "l"(p));
        return r;
}

__device__ __forceinline__
uint load4_last(__global ulong *p,uint offset){
	p=(__global ulong *)((__global char *)p + offset); 
        uint r;
        asm volatile ( "ld.global.cs.nc.b32  %0, [%1];\n\t" : "=r"(r) : "l"(p));
        return r;
}
__device__ __forceinline__
uint load4(__global ulong *p,uint offset){
	p=(__global ulong *)((__global char *)p + offset); 
        uint r;
        asm volatile ( "ld.global.cs.nc.b32  %0, [%1];\n\t" : "=r"(r) : "l"(p));
        return r;
}


__device__ __forceinline__
void trigger_err(){
	load8_last((__global ulong *)-1,0);
}



// Round 1
__device__ __forceinline__
uint xor_and_store1(uint round, __global char *ht_dst, uint x_row,
        uint slot_a, uint slot_b, __global ulong *a, __global ulong *b,
        __global uint *rowCounters){
	
	ulong xi0, xi1, xi2,xi3;
	uint _row;
	uint row;
	__global char       *p;
        uint                cnt;
//LOAD

	ulong2 loada,loadb;
	xi0 = load8(a++,0) ^ load8(b++,0);
//	loada = *(__global ulong2 *)a;
	loada = load16l(a,0);
	loadb = load16l(b,0);
	xi1 = loada.x ^ loadb.x;
	xi2 = loada.y ^ loadb.y;


/*
	xi0 = *(a++) ^ *(b++);
	xi1 = *(a++) ^ *(b++);
	xi2 = *a ^ *b;
	xi3 = 0;
*/
//
	uint i = ENCODE_INPUTS(x_row, slot_a, slot_b);
	

	//256bit shift


	asm("{ .reg .b16 a0,a1,a2,a3,b0,b1,b2,b3,c0,c1,c2,c3; \n\t"
	"mov.b64 {a0,a1,a2,a3}, %4;\n\t"
	"mov.b64 {b0,b1,b2,b3}, %5;\n\t"
	"mov.b64 {c0,c1,c2,c3}, %6;\n\t"

	"mov.b64 %0, {a1,a2,a3,b0};\n\t"
	"mov.b64 %1, {b1,b2,b3,c0};\n\t"
	"mov.b64 %2, {c1,c2,c3,0};\n\t"
	"mov.b32 %3, {a0,a1};\n\t"
	"}\n" : "=l"(xi0), "=l"(xi1), "=l" (xi2), "=r"(_row): "l"(xi0), "l"(xi1), "l"(xi2));


//      row = get_row_nr_4((uint)xi0,round);	
	row = get_row_nr_4(_row,round);

//        xi0 = (xi0 >> 16) | (xi1 << (64 - 16));
//        xi1 = (xi1 >> 16) | (xi2 << (64 - 16));
//        xi2 = (xi2 >> 16);
	
//
	
    p = ht_dst + row * NR_SLOTS * SLOT_LEN;
    uint rowIdx = row/ROWS_PER_UINT;
    uint rowOffset = BITS_PER_ROW*(row%ROWS_PER_UINT);
    uint xcnt = atomic_add(rowCounters + rowIdx, 1 << rowOffset);
    xcnt = (xcnt >> rowOffset) & ROW_MASK;
    cnt = xcnt;
    if (cnt >= NR_SLOTS)
      {
	// avoid overflows
	atomic_sub(rowCounters + rowIdx, 1 << rowOffset);
	return 1;
      }
    __global char       *pp = p + cnt * SLOT_LEN;
    p = pp + xi_offset_for_round(round);
//

//STORE
//        *(__global uint *)(p - 4) = i;
//        *(__global ulong *)(p + 0) = xi0;
//	*(__global ulong *)(p + 8) = xi1;
//	*(__global ulong *)(p + 16) = xi2;


	ulong2 store0;
	ulong2 store1;
	nv32to64(store0.x,0,i);
	store0.y=xi0;
//	*(__global ulong2 *)(pp)=store0;
	store_ulong2(pp,store0);
	store1.x=xi1;
	store1.y=xi2;
//	*(__global ulong2 *)(pp+16)=store1;
	store_ulong2(pp+16,store1);
return 0;
}



// Round 2
__device__ __forceinline__
uint xor_and_store2(uint round, __global char *ht_dst, uint x_row,
        uint slot_a, uint slot_b, __global ulong *a, __global ulong *b,
        __global uint *rowCounters){
	
	ulong xi0, xi1, xi2,xi3;

	uint _row;
	uint row;
	__global char       *p;
        uint                cnt;
//LOAD
	ulong2 loada,loadb;
	xi0 = load8(a++,0) ^ load8(b++,0);
	loada = load16l(a,0);
	loadb = load16l(b,0);
	xi1 = loada.x ^ loadb.x;
	xi2 = loada.y ^ loadb.y;


/*
	xi0 = *(a++) ^ *(b++);
	xi1 = *(a++) ^ *(b++);
	xi2 = *a ^ *b;
	xi3 = 0;
*/
//
	uint i = ENCODE_INPUTS(x_row, slot_a, slot_b);
	

	//256bit shift



//7 op asm32 4 op + 3 op devectorize

	uint _xi0l,_xi0h,_xi1l,_xi1h,_xi2l,_xi2h;
	asm("{\n\t"
			".reg .b32 a0,a1,b0,b1,c0,c1; \n\t"
			"mov.b64 {a0,a1}, %6;\n\t"
        	        "mov.b64 {b0,b1}, %7;\n\t"
                        "mov.b64 {c0,c1}, %8;\n\t"
			
			"shr.b32 %5,a0,8;\n\t"
                        "shf.r.clamp.b32 %0,a0,a1,24; \n\t"
                        "shf.r.clamp.b32 %1,a1,b0,24; \n\t"
                        "shf.r.clamp.b32 %2,b0,b1,24; \n\t"
			"shf.r.clamp.b32 %3,b1,c0,24; \n\t"
			"shf.r.clamp.b32 %4,c0,c1,24; \n\t"

                        "}\n\t"
                        : "=r"(_xi0l), "=r"(_xi0h),"=r"(_xi1l), "=r"(_xi1h), "=r"(_xi2l), "=r"(_row) :  
			"l"(xi0), "l"(xi1), "l"(xi2));

	row = get_row_nr_4(_row,round);

//	xi0 = (xi0 >> 24) | (xi1 << (64 - 24));
//        xi1 = (xi1 >> 24) | (xi2 << (64 - 24));
//        xi2 = (xi2 >> 24);
//

    p = ht_dst + row * NR_SLOTS * SLOT_LEN;
    uint rowIdx = row/ROWS_PER_UINT;
    uint rowOffset = BITS_PER_ROW*(row%ROWS_PER_UINT);
    uint xcnt = atomic_add(rowCounters + rowIdx, 1 << rowOffset);
    xcnt = (xcnt >> rowOffset) & ROW_MASK;
    cnt = xcnt;
    if (cnt >= NR_SLOTS)
      {
	// avoid overflows
//	*a+=load8_last((__global ulong *)-1);
	atomic_sub(rowCounters + rowIdx, 1 << rowOffset);
	return 1;
      }
    __global char       *pp = p + cnt * SLOT_LEN;
    p = pp + xi_offset_for_round(round);
//

//STORE 11 op, asm 9 op, or 6op 32bit

/*
        ulong s0;
	ulong2 store0;
	nv32to64(s0,i,_xi0l);
	nv32to64(store0.x,_xi0h,_xi1l);
	nv32to64(store0.y,_xi1h,_xi2l);
	*(__global ulong *)(p - 4)=s0;
	*(__global ulong2 *)(p + 4)=store0;
*/

	uint2 s0;
	s0.x=i;
	s0.y=_xi0l;
        uint4 store0;
	store0.x=_xi0h;
	store0.y=_xi1l;
	store0.z=_xi1h;
	store0.w=_xi2l;
//	*(__global uint2 *)(p - 4)=s0;
	store_uint2(p-4, s0);
//        *(__global uint4 *)(p + 4)=store0;
	store_uint4(p+4,store0);
/*
	*(__global uint *)(p - 4) = i;
	*(__global uint *)(p + 0) = xi0;
	*(__global ulong *)(p + 4) = (xi0 >> 32) | (xi1 << 32);
	*(__global ulong *)(p + 12) = (xi1 >> 32) | (xi2 << 32);
*/
return 0;
}



//Round3
__device__ __forceinline__
uint xor_and_store3(uint round, __global char *ht_dst, uint x_row,
        uint slot_a, uint slot_b, __global ulong *a, __global ulong *b,
        __global uint *rowCounters){
	
//	ulong xi0, xi1, xi2,xi3;
	uint _row;
	uint row;
	__global char       *p;
        uint                cnt;
//LOAD
	uint xi0l,xi0h,xi1l,xi1h,xi2l;
	xi0l = load4(a,0) ^ load4(b,0);
	
	if(!xi0l )
		return 0;


	ulong load1,load2;
	load1 = load8(a , 4) ^ load8(b , 4);
	load2 = load8_last(a , 12) ^ load8_last(b , 12);
	nv64to32(xi0h,xi1l,load1);
	nv64to32(xi1h,xi2l,load2);

//     if(!xi0l )
//	*a+=load8_last((__global ulong *)-1);
	// xor 20 bytes
//	xi0 = half_aligned_long(a, 0) ^ half_aligned_long(b, 0);
//	xi1 = half_aligned_long(a, 8) ^ half_aligned_long(b, 8);
//	xi2 = well_aligned_int(a, 16) ^ well_aligned_int(b, 16);
//	ulong2 loada;
//	ulong2 loadb;
	

//
	uint i = ENCODE_INPUTS(x_row, slot_a, slot_b);
	



	row = get_row_nr_4(xi0l,round);	


	uint _xi0l,_xi0h,_xi1l,_xi1h;
	asm("{\n\t"
                        "shf.r.clamp.b32 %0,%4,%5,16; \n\t"
                        "shf.r.clamp.b32 %1,%5,%6,16; \n\t"
                        "shf.r.clamp.b32 %2,%6,%7,16; \n\t"
                        "shf.r.clamp.b32 %3,%7,%8,16; \n\t"
                        "}\n\t"
                        : "=r"(_xi0l), "=r"(_xi0h),"=r"(_xi1l), "=r"(_xi1h):  
                        "r"(xi0l), "r"(xi0h),"r"(xi1l), "r"(xi1h) , "r"(xi2l));




//        xi0 = (xi0 >> 16) | (xi1 << (64 - 16));
//        xi1 = (xi1 >> 16) | (xi2 << (64 - 16));
//        xi2 = (xi2 >> 16);
	
//
	
    p = ht_dst + row * NR_SLOTS * SLOT_LEN;
    uint rowIdx = row/ROWS_PER_UINT;
    uint rowOffset = BITS_PER_ROW*(row%ROWS_PER_UINT);
    uint xcnt = atomic_add(rowCounters + rowIdx, 1 << rowOffset);
    xcnt = (xcnt >> rowOffset) & ROW_MASK;
    cnt = xcnt;
    if (cnt >= NR_SLOTS)
      {
	// avoid overflows
//	*a+=load8_last((__global ulong *)-1);
	atomic_sub(rowCounters + rowIdx, 1 << rowOffset);
	return 1;
      }
    __global char       *pp = p + cnt * SLOT_LEN;
    p = pp + xi_offset_for_round(round);
//

//STORE
	

	

	ulong store0,store1;
       nv32to64(store0,i,_xi0l);
	nv32to64(store1,_xi0h,_xi1l);

//        *(__global ulong *)(p - 4) = store0;
	store8(p - 4,store0);
//        *(__global ulong *)(p + 4) = store1;
	store8(p + 4,store1);
//        *(__global uint *)(p + 12) = _xi1h;
	store4(p + 12,_xi1h);

/*
	 *(__global uint *)(p - 4) = i;
		// store 16 bytes
	*(__global uint *)(p + 0) = xi0;
	*(__global ulong *)(p + 4) = (xi0 >> 32) | (xi1 << 32);
	*(__global uint *)(p + 12) = (xi1 >> 32);
*/
return 0;
}



// Round 4
__device__ __forceinline__
uint xor_and_store4(uint round, __global char *ht_dst, uint x_row,
        uint slot_a, uint slot_b, __global ulong *a, __global ulong *b,
        __global uint *rowCounters){
	
	ulong xi0, xi1, xi2,xi3;
	uint _row;
	uint row;
	__global char       *p;
        uint                cnt;
//LOAD

//	xi0 = half_aligned_long(a, 0) ^ half_aligned_long(b, 0);
//	xi1 = half_aligned_long(a, 8) ^ half_aligned_long(b, 8);
	

	uint xi0l,xi0h,xi1l,xi1h;
	xi0l = load4(a, 0) ^ load4(b, 0);
	        if(!xi0l )
                return 0;
	xi0h = load4(a, 4) ^ load4(b, 4);
	xi1l = load4(a, 8) ^ load4(b, 8);
	xi1h = load4_last(a, 12) ^ load4_last(b, 12);


//	xi2 = 0;

//
	uint i = ENCODE_INPUTS(x_row, slot_a, slot_b);
	

//256bit shift

	uint _xi0l,_xi0h,_xi1l,_xi1h,_xi2l,_xi2h;
	asm("{\n\t"
                        "shf.r.clamp.b32 %0,%4,%5,24; \n\t"
                        "shf.r.clamp.b32 %1,%5,%6,24; \n\t"
                        "shf.r.clamp.b32 %2,%6,%7,24; \n\t"
			"shr.b32         %3,%7,24; \n\t"
                        "}\n\t"
                        : "=r"(_xi0l), "=r"(_xi0h),"=r"(_xi1l), "=r"(_xi1h):  
			"r"(xi0l), "r"(xi0h), "r"(xi1l), "r"(xi1h));

	row = get_row_nr_4(xi0l >> 8,round);

//            xi0 = (xi0 >> 8) | (xi1 << (64 - 8));
//	    xi1 = (xi1 >> 8);

      //row = get_row_nr_4((uint)xi0,round);	
//	row = get_row_nr_4(_row,round);

 //       xi0 = (xi0 >> 16) | (xi1 << (64 - 16));
 //       xi1 = (xi1 >> 16) | (xi2 << (64 - 16));
 //       xi2 = (xi2 >> 16);
	
//
	
    p = ht_dst + row * NR_SLOTS * SLOT_LEN;
    uint rowIdx = row/ROWS_PER_UINT;
    uint rowOffset = BITS_PER_ROW*(row%ROWS_PER_UINT);
    uint xcnt = atomic_add(rowCounters + rowIdx, 1 << rowOffset);
    xcnt = (xcnt >> rowOffset) & ROW_MASK;
    cnt = xcnt;
    if (cnt >= NR_SLOTS)
      {
	// avoid overflows
	atomic_sub(rowCounters + rowIdx, 1 << rowOffset);
	return 1;
      }
    __global char       *pp = p + cnt * SLOT_LEN;
    p = pp + xi_offset_for_round(round);
//

//STORE

// *(__global uint *)(p - 4) = i;
store4(p-4,i);
// *(__global ulong *)(p + 0) = xi0;
// *(__global ulong *)(p + 8) = xi1;
uint4 store;
	store.x=_xi0l;
	store.y=_xi0h;
	store.z=_xi1l;
	store.w=_xi1h;
// *(__global uint4 *)(p + 0) = store;
store_uint4(p + 0, store);
return 0;
}


// Round 5
__device__ __forceinline__
uint xor_and_store5(uint round, __global char *ht_dst, uint x_row,
        uint slot_a, uint slot_b, __global ulong *a, __global ulong *b,
        __global uint *rowCounters){
	
	ulong xi0, xi1, xi2,xi3;
	uint _row;
	uint row;
	__global char       *p;
        uint                cnt;
//LOAD

//	xi0 = half_aligned_long(a, 0) ^ half_aligned_long(b, 0);
//	xi1 = half_aligned_long(a, 8) ^ half_aligned_long(b, 8);
	

	uint xi0l,xi0h,xi1l,xi1h;
	xi0l = load4(a, 0) ^ load4(b, 0);
	        if(!xi0l )
                return 0;
	xi0h = load4(a, 4) ^ load4(b, 4);
	xi1l = load4(a, 8) ^ load4(b, 8);
	xi1h = load4_last(a, 12) ^ load4_last(b, 12);


//	xi2 = 0;

//
	uint i = ENCODE_INPUTS(x_row, slot_a, slot_b);
	

//256bit shift

	uint _xi0l,_xi0h,_xi1l,_xi1h,_xi2l,_xi2h;
	asm("{\n\t"
                        "shf.r.clamp.b32 %0,%4,%5,16; \n\t"
                        "shf.r.clamp.b32 %1,%5,%6,16; \n\t"
                        "shf.r.clamp.b32 %2,%6,%7,16; \n\t"
			"shr.b32         %3,%7,16; \n\t"
                        "}\n\t"
                        : "=r"(_xi0l), "=r"(_xi0h),"=r"(_xi1l), "=r"(_xi1h):  
			"r"(xi0l), "r"(xi0h), "r"(xi1l), "r"(xi1h));

	row = get_row_nr_4(xi0l,round);

//            xi0 = (xi0 >> 8) | (xi1 << (64 - 8));
//	    xi1 = (xi1 >> 8);

      //row = get_row_nr_4((uint)xi0,round);	
//	row = get_row_nr_4(_row,round);

 //       xi0 = (xi0 >> 16) | (xi1 << (64 - 16));
 //       xi1 = (xi1 >> 16) | (xi2 << (64 - 16));
 //       xi2 = (xi2 >> 16);
	
//
	
    p = ht_dst + row * NR_SLOTS * SLOT_LEN;
    uint rowIdx = row/ROWS_PER_UINT;
    uint rowOffset = BITS_PER_ROW*(row%ROWS_PER_UINT);
    uint xcnt = atomic_add(rowCounters + rowIdx, 1 << rowOffset);
    xcnt = (xcnt >> rowOffset) & ROW_MASK;
    cnt = xcnt;
    if (cnt >= NR_SLOTS)
      {
	// avoid overflows
	atomic_sub(rowCounters + rowIdx, 1 << rowOffset);
	return 1;
      }
    __global char       *pp = p + cnt * SLOT_LEN;
    p = pp + xi_offset_for_round(round);
//

//STORE

// *(__global uint *)(p - 4) = i;
store4(p-4,i);
// *(__global ulong *)(p + 0) = xi0;
// *(__global ulong *)(p + 8) = xi1;
uint4 store;
	store.x=_xi0l;
	store.y=_xi0h;
	store.z=_xi1l;
	store.w=_xi1h;
// *(__global uint4 *)(p + 0) = store;
store_uint4(p + 0, store);
return 0;
}




// Round 6
__device__ __forceinline__
uint xor_and_store6(uint round, __global char *ht_dst, uint x_row,
        uint slot_a, uint slot_b, __global ulong *a, __global ulong *b,
        __global uint *rowCounters){
	
	ulong xi0, xi1, xi2,xi3;
	uint _row;
	uint row;
	__global char       *p;
        uint                cnt;
//LOAD
	uint xi0l,xi0h,xi1l;

	xi0 = load8(a++,0) ^ load8(b++,0);

	if(!xi0 )
                return 0;
	xi1l = load4_last(a,0) ^ load4_last(b,0);
	
	nv64to32(xi0l,xi0h,xi0);

//	xi0 = (xi0 >> 8) | (xi1 << (64 - 8));
//	xi1 = (xi1 >> 8);

//	xi2 = 0;

//
	uint i = ENCODE_INPUTS(x_row, slot_a, slot_b);
	

//256bit shift

	uint _xi0l,_xi0h,_xi1l,_xi1h,_xi2l,_xi2h;
	asm("{\n\t"
                        "shf.r.clamp.b32 %0,%3,%4,24; \n\t"
                        "shf.r.clamp.b32 %1,%4,%5,24; \n\t"
			"shr.b32         %2,%5,24; \n\t"
                        "}\n\t"
                        : "=r"(_xi0l), "=r"(_xi0h),"=r"(_xi1l):  
			"r"(xi0l), "r"(xi0h), "r"(xi1l));

	row = get_row_nr_4(xi0l >> 8,round);

	
//
	
    p = ht_dst + row * NR_SLOTS * SLOT_LEN;
    uint rowIdx = row/ROWS_PER_UINT;
    uint rowOffset = BITS_PER_ROW*(row%ROWS_PER_UINT);
    uint xcnt = atomic_add(rowCounters + rowIdx, 1 << rowOffset);
    xcnt = (xcnt >> rowOffset) & ROW_MASK;
    cnt = xcnt;
    if (cnt >= NR_SLOTS)
      {
	// avoid overflows
	atomic_sub(rowCounters + rowIdx, 1 << rowOffset);
	return 1;
      }
    __global char       *pp = p + cnt * SLOT_LEN;
    p = pp + xi_offset_for_round(round);
//

//STORE
	
//	*(__global uint *)(p - 4) = i;
	ulong store;
	nv32to64(store,i,_xi0l);
	store8(p - 4,store);
	// *(__global ulong *)(p - 4)= store;
//	*(__global uint *)(p + 0) = _xi0l;
//	*(__global uint *)(p + 4) = _xi0h;
	store4(p+4,_xi0h);
return 0;
}


// Round 7
__device__ __forceinline__
uint xor_and_store7(uint round, __global char *ht_dst, uint x_row,
        uint slot_a, uint slot_b, __global ulong *a, __global ulong *b,
        __global uint *rowCounters){
	
	ulong xi0, xi1, xi2,xi3;
	uint _row;
	uint row;
	__global char       *p;
        uint                cnt;
//LOAD

	uint xi0l,xi0h;
	xi0l = load4(a, 0) ^ load4(b, 0);
	        if(!xi0l )
                return 0;
	xi0h = load4_last(a, 4) ^ load4_last(b, 4);
//
	uint i = ENCODE_INPUTS(x_row, slot_a, slot_b);
	

//256bit shift


	row = get_row_nr_4(xi0l,round);

	uint _xi0l,_xi0h;
	asm("{\n\t"
                        "shf.r.clamp.b32 %0,%2,%3,16; \n\t"
			"shr.b32         %1,%3,16; \n\t"
                        "}\n\t"
                        : "=r"(_xi0l), "=r"(_xi0h):  
			"r"(xi0l), "r"(xi0h));
//
	
    p = ht_dst + row * NR_SLOTS * SLOT_LEN;
    uint rowIdx = row/ROWS_PER_UINT;
    uint rowOffset = BITS_PER_ROW*(row%ROWS_PER_UINT);
    uint xcnt = atomic_add(rowCounters + rowIdx, 1 << rowOffset);
    xcnt = (xcnt >> rowOffset) & ROW_MASK;
    cnt = xcnt;
    if (cnt >= NR_SLOTS)
      {
	// avoid overflows
	atomic_sub(rowCounters + rowIdx, 1 << rowOffset);
	return 1;
      }
    __global char       *pp = p + cnt * SLOT_LEN;
    p = pp + xi_offset_for_round(round);
//

//STORE
	
	uint2 store;
	store.x=i;
	store.y=_xi0l;
//	*(__global uint2 *)(p - 4) = store;
	store_uint2(p-4,store);
//	*(__global uint *)(p + 0) = _xi0l;
//	*(__global uint *)(p + 4) = _xi0h;
	store4(p + 4 , _xi0h);
return 0;
}

// Round 8
__device__ __forceinline__
uint xor_and_store8(uint round, __global char *ht_dst, uint x_row,
        uint slot_a, uint slot_b, __global ulong *a, __global ulong *b,
        __global uint *rowCounters){
	
	ulong xi0, xi1, xi2,xi3;
	uint _row;
	uint row;
	__global char       *p;
        uint                cnt;
//LOAD

	uint xi0l,xi0h;
	xi0l = load4(a, 0) ^ load4(b, 0);
	        if(!xi0l )
                return 0;
	xi0h = load4_last(a, 4) ^ load4_last(b, 4);
//
	uint i = ENCODE_INPUTS(x_row, slot_a, slot_b);
	

//256bit shift


	row = get_row_nr_4(xi0l >> 8,round);

	
	uint _xi0l,_xi0h,_xi1l,_xi1h,_xi2l,_xi2h;
	asm("{\n\t"
                        "shf.r.clamp.b32 %0,%1,%2,24; \n\t"
                        "}\n\t"
                        : "=r"(_xi0l):  
			"r"(xi0l), "r"(xi0h));

//
	
    p = ht_dst + row * NR_SLOTS * SLOT_LEN;
    uint rowIdx = row/ROWS_PER_UINT;
    uint rowOffset = BITS_PER_ROW*(row%ROWS_PER_UINT);
    uint xcnt = atomic_add(rowCounters + rowIdx, 1 << rowOffset);
    xcnt = (xcnt >> rowOffset) & ROW_MASK;
    cnt = xcnt;
    if (cnt >= NR_SLOTS)
      {
	// avoid overflows
	atomic_sub(rowCounters + rowIdx, 1 << rowOffset);
	return 1;
      }
    __global char       *pp = p + cnt * SLOT_LEN;
    p = pp + xi_offset_for_round(round);
//

//STORE
	
//	uint2 store;
//	store.x=i;
//	store.y=_xi0l;

//	*(__global uint *)(p - 4) = i;
//	*(__global uint *)(p + 0) = _xi0l;
	store4(p-4, i);
	store4(p+0, _xi0l);
return 0;
}

__device__ __forceinline__
uint xor_and_store(uint round, __global char *ht_dst, uint row,
	uint slot_a, uint slot_b, __global ulong *a, __global ulong *b,
	__global uint *rowCounters)
{

	if(round == 1)
		return xor_and_store1(round,ht_dst,row,slot_a,slot_b,a,b,rowCounters);
	else if(round == 2)
                return xor_and_store2(round,ht_dst,row,slot_a,slot_b,a,b,rowCounters);
	else if(round == 3)
		return xor_and_store3(round,ht_dst,row,slot_a,slot_b,a,b,rowCounters);
	else if(round == 4)
                return xor_and_store4(round,ht_dst,row,slot_a,slot_b,a,b,rowCounters);
	else if(round == 5)
                return xor_and_store5(round,ht_dst,row,slot_a,slot_b,a,b,rowCounters);
	else if(round == 6)
                return xor_and_store6(round,ht_dst,row,slot_a,slot_b,a,b,rowCounters);
        else if(round == 7)
                return xor_and_store7(round,ht_dst,row,slot_a,slot_b,a,b,rowCounters);
        else if(round == 8)
                return xor_and_store8(round,ht_dst,row,slot_a,slot_b,a,b,rowCounters);
}

/*
** Execute one Equihash round. Read from ht_src, XOR colliding pairs of Xi,
** store them in ht_dst.
*/
__device__ __forceinline__
void equihash_round(uint round,
  __global char *ht_src,
  __global char *ht_dst,
  __global uint *debug,
  uint *collisionsData,
  uint *collisionsNum,
  __global uint *rowCountersSrc,
  __global uint *rowCountersDst)
{
    uint globalTid = get_global_id(0) / THREADS_PER_ROW;
    uint localRowIdx = get_local_id(0) / THREADS_PER_ROW;
    uint localTid = get_local_id(0) % THREADS_PER_ROW;
    __local uint slotCountersData[COLLISION_TYPES_NUM*ROWS_PER_WORKGROUP];
    __local ushort slotsData[COLLISION_TYPES_NUM*COLLISION_BUFFER_SIZE*ROWS_PER_WORKGROUP];
    
    uint *slotCounters = &slotCountersData[COLLISION_TYPES_NUM*localRowIdx];
    ushort *slots = &slotsData[COLLISION_TYPES_NUM*COLLISION_BUFFER_SIZE*localRowIdx];
    
    __global char *p;
    uint    cnt;
    uchar   mask;
    uint    shift;
    uint    i, j;
    // NR_SLOTS is already oversized (by a factor of OVERHEAD), but we want to
    // make it even larger
    uint    n;
    uint    dropped_coll = 0;
    uint    dropped_stor = 0;
    __global ulong  *a, *b;
    uint    xi_offset;
    // read first words of Xi from the previous (round - 1) hash table
    xi_offset = xi_offset_for_round(round - 1);
    // the mask is also computed to read data from the previous round
#if NR_ROWS_LOG <= 16
    mask = ((!(round % 2)) ? 0x0f : 0xf0);
    shift = ((!(round % 2)) ? 0 : 4);
#elif NR_ROWS_LOG == 18
    mask = ((!(round % 2)) ? 0x03 : 0x30);
    shift = ((!(round % 2)) ? 0 : 4);    
#elif NR_ROWS_LOG == 19
    mask = ((!(round % 2)) ? 0x01 : 0x10);
    shift = ((!(round % 2)) ? 0 : 4);    
#elif NR_ROWS_LOG == 20
    mask = 0; /* we can vastly simplify the code below */
    shift = 0;
#else
#error "unsupported NR_ROWS_LOG"
#endif    
    
  for (uint chunk = 0; chunk < THREADS_PER_ROW; chunk++) {
    uint tid = globalTid + NR_ROWS/THREADS_PER_ROW*chunk;
    uint gid = tid & ~(ROWS_PER_WORKGROUP - 1);
    
    uint rowIdx = tid/ROWS_PER_UINT;
    uint rowOffset = BITS_PER_ROW*(tid%ROWS_PER_UINT);
    cnt = (rowCountersSrc[rowIdx] >> rowOffset) & ROW_MASK;
    cnt = min(cnt, (uint)NR_SLOTS); // handle possible overflow in prev. round

    *collisionsNum = 0;
    p = (ht_src + tid * NR_SLOTS * SLOT_LEN);
    p += xi_offset;
    p += SLOT_LEN*localTid;

    for (i = get_local_id(0); i < COLLISION_TYPES_NUM*ROWS_PER_WORKGROUP; i += get_local_size(0))
      slotCountersData[i] = 0;
    barrier(CLK_LOCAL_MEM_FENCE);
    
    for (i = localTid; i < cnt; i += THREADS_PER_ROW, p += SLOT_LEN*THREADS_PER_ROW) {
      uchar x = ((*(__global uchar *)p) & mask) >> shift;
      uint slotIdx = atomic_inc(&slotCounters[x]);
      slotIdx = min(slotIdx, COLLISION_BUFFER_SIZE-1);
      slots[COLLISION_BUFFER_SIZE*x+slotIdx] = i;
    }
    
    barrier(CLK_LOCAL_MEM_FENCE);
    const uint ct_groupsize = max(1u, THREADS_PER_ROW / COLLISION_TYPES_NUM);
    for (uint collTypeIdx = localTid / ct_groupsize; collTypeIdx < COLLISION_TYPES_NUM; collTypeIdx += THREADS_PER_ROW / ct_groupsize) {
      const uint N = min((uint)slotCounters[collTypeIdx], COLLISION_BUFFER_SIZE);
      for (uint i = 0; i < N; i++) {
        uint collision = (localRowIdx << 24) | (slots[COLLISION_BUFFER_SIZE*collTypeIdx+i] << 12);
        for (uint j = i + 1 + localTid % ct_groupsize; j < N; j += ct_groupsize) {
          uint index = atomic_inc(collisionsNum);
          index = min(index, (uint)(LDS_COLL_SIZE-1));
          collisionsData[index] = collision | slots[COLLISION_BUFFER_SIZE*collTypeIdx+j];
        }
      }
    }

    barrier(CLK_LOCAL_MEM_FENCE);
    uint totalCollisions = *collisionsNum;
    totalCollisions = min(totalCollisions, (uint)LDS_COLL_SIZE);
    for (uint index = get_local_id(0); index < totalCollisions; index += get_local_size(0))
      {
  uint collision = collisionsData[index];
  uint collisionThreadId = gid + (collision >> 24);
  uint i = (collision >> 12) & 0xFFF;
  uint j = collision & 0xFFF;
  __global char *ptr = ht_src + collisionThreadId * NR_SLOTS * SLOT_LEN +
      xi_offset;
  a = (__global ulong *)(ptr + i * SLOT_LEN);
  b = (__global ulong *)(ptr + j * SLOT_LEN);
  dropped_stor += xor_and_store(round, ht_dst, collisionThreadId, i, j,
    a, b, rowCountersDst);
      }
  }
      
#ifdef ENABLE_DEBUG
    debug[tid * 2] = dropped_coll;
    debug[tid * 2 + 1] = dropped_stor;
#endif
}


/*
** This defines kernel_round1, kernel_round2, ..., kernel_round7.
*/
#define KERNEL_ROUND(N) \
__global__ void kernel_round ## N(__global char *ht_src, __global char *ht_dst, \
	__global uint *rowCountersSrc, __global uint *rowCountersDst, \
       	__global uint *debug) \
{ \
    __local uint    collisionsData[LDS_COLL_SIZE]; \
    __local uint    collisionsNum; \
    equihash_round(N, ht_src, ht_dst, debug, collisionsData, \
	    &collisionsNum, rowCountersSrc, rowCountersDst); \
}

KERNEL_ROUND(1)
KERNEL_ROUND(2)
KERNEL_ROUND(3)
KERNEL_ROUND(4)
KERNEL_ROUND(5)
KERNEL_ROUND(6)
KERNEL_ROUND(7)


// kernel_round8 takes an extra argument, "sols"
__global__ 
void kernel_round8(__global char *ht_src, __global char *ht_dst,
	__global uint *rowCountersSrc, __global uint *rowCountersDst,
	__global uint *debug, __global sols_t *sols)
{
    uint		tid = get_global_id(0);
    __local uint	collisionsData[LDS_COLL_SIZE];
    __local uint	collisionsNum;
    equihash_round(8, ht_src, ht_dst, debug, collisionsData,
	    &collisionsNum, rowCountersSrc, rowCountersDst);
    if (!tid)
	sols->nr = sols->likely_invalids = 0;
}


__device__ __forceinline__
uint expand_ref(__global char *ht, uint xi_offset, uint row, uint slot)
{
    return *(__global uint *)(ht + row * NR_SLOTS * SLOT_LEN +
	    slot * SLOT_LEN + xi_offset - 4);
}


__device__ __forceinline__
uint expand_refs(uint *ins, uint nr_inputs, __global char **htabs,
	uint round)
{
    __global char	*ht = htabs[round % 2];
    uint		i = nr_inputs - 1;
    uint		j = nr_inputs * 2 - 1;
    uint		xi_offset = xi_offset_for_round(round);
    int			dup_to_watch = -1;
    do
      {
	ins[j] = expand_ref(ht, xi_offset,
		DECODE_ROW(ins[i]), DECODE_SLOT1(ins[i]));
	ins[j - 1] = expand_ref(ht, xi_offset,
		DECODE_ROW(ins[i]), DECODE_SLOT0(ins[i]));
	if (!round)
	  {
	    if (dup_to_watch == -1)
		dup_to_watch = ins[j];
	    else if (ins[j] == dup_to_watch || ins[j - 1] == dup_to_watch)
		return 0;
	  }
	if (!i)
	    break ;
	i--;
	j -= 2;
      }
    while (1);
    return 1;
}


/*
** Verify if a potential solution is in fact valid.
*/
__device__ __forceinline__
void potential_sol(__global char **htabs, __global sols_t *sols,
	uint ref0, uint ref1)
{
    uint	nr_values;
    uint	values_tmp[(1 << PARAM_K)];
    uint	sol_i;
    uint	i;
    nr_values = 0;
    values_tmp[nr_values++] = ref0;
    values_tmp[nr_values++] = ref1;
    uint round = PARAM_K - 1;
    do
      {
	round--;
	if (!expand_refs(values_tmp, nr_values, htabs, round))
	    return ;
	nr_values *= 2;
      }
    while (round > 0);
    // solution appears valid, copy it to sols
    sol_i = atomic_inc(&sols->nr);
    if (sol_i >= MAX_SOLS)
	return ;
    for (i = 0; i < (1 << PARAM_K); i++)
	sols->values[sol_i][i] = values_tmp[i];
    sols->valid[sol_i] = 1;
}


/*
** Scan the hash tables to find Equihash solutions.
*/
__global__
void kernel_sols(__global char *ht0, __global char *ht1, __global sols_t *sols,
	__global uint *rowCountersSrc, __global uint *rowCountersDst)
{
    __local uint counters[THRD/THREADS_PER_ROW];
    __local uint refs[NR_SLOTS*(THRD/THREADS_PER_ROW)];
    __local uint data[NR_SLOTS*(THRD/THREADS_PER_ROW)];
    __local uint collisionsNum;
    __local ulong collisions[THRD*4];

    uint globalTid = get_global_id(0) / THREADS_PER_ROW;
    uint localTid = get_local_id(0) / THREADS_PER_ROW;
    uint localGroupId = get_local_id(0) % THREADS_PER_ROW;
    uint *refsPtr = &refs[NR_SLOTS*localTid];
    uint *dataPtr = &data[NR_SLOTS*localTid];    
    
    __global char	*htabs[2] = { ht0, ht1 };
//    __global char	*hcounters[2] = { rowCountersSrc, rowCountersDst };
    uint		ht_i = (PARAM_K - 1) % 2; // table filled at last round
    uint		cnt;
    uint		xi_offset = xi_offset_for_round(PARAM_K - 1);
    uint		i, j;
    __global char	*p;
    uint		ref_i, ref_j;
    // it's ok for the collisions array to be so small, as if it fills up
    // the potential solutions are likely invalid (many duplicate inputs)
//     ulong		collisions;
#if NR_ROWS_LOG >= 16 && NR_ROWS_LOG <= 20
    // in the final hash table, we are looking for a match on both the bits
    // part of the previous PREFIX colliding bits, and the last PREFIX bits.
    uint		mask = 0xffffff;
#else
#error "unsupported NR_ROWS_LOG"
#endif
    
  collisionsNum = 0;    
  
  for (uint chunk = 0; chunk < THREADS_PER_ROW; chunk++) {
    uint tid = globalTid + NR_ROWS/THREADS_PER_ROW*chunk;
    p = htabs[ht_i] + tid * NR_SLOTS * SLOT_LEN;
    uint rowIdx = tid/ROWS_PER_UINT;
    uint rowOffset = BITS_PER_ROW*(tid%ROWS_PER_UINT);
    cnt = (rowCountersSrc[rowIdx] >> rowOffset) & ROW_MASK;
    cnt = min(cnt, (uint)NR_SLOTS); // handle possible overflow in last round
    p += xi_offset;
    p += SLOT_LEN*localGroupId;

    for (i = get_local_id(0); i < THRD/THREADS_PER_ROW; i += get_local_size(0))
      counters[i] = 0;
    for (i = localGroupId; i < cnt; i += THREADS_PER_ROW, p += SLOT_LEN*THREADS_PER_ROW) {
      refsPtr[i] = *(__global uint *)(p - 4);
      dataPtr[i] = (*(__global uint *)p) & mask;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    for (i = 0; i < cnt; i++)
      {
	uint a_data = dataPtr[i];
	ref_i = refsPtr[i];
	for (j = i + 1 + localGroupId; j < cnt; j += THREADS_PER_ROW)
	  {
	    if (a_data == dataPtr[j])
	      {
          if (atomic_inc(&counters[localTid]) == 0)
		        collisions[atomic_inc(&collisionsNum)] = ((ulong)ref_i << 32) | refsPtr[j];
          goto part2;          
	      }
	  }
      }

part2:
    continue;
  }
  
    barrier(CLK_LOCAL_MEM_FENCE);
    uint totalCollisions = collisionsNum;
    if (get_local_id(0) < totalCollisions) {
      ulong coll = collisions[get_local_id(0)];
      potential_sol(htabs, sols, coll >> 32, coll & 0xffffffff);
    }
}


struct __align__(64) c_context {
	char* buf_ht[2], *buf_sols, *buf_dbg;
	char *rowCounters[2];
	sols_t	*sols;
	u32 nthreads;
	size_t global_ws;


	c_context(const u32 n_threads) {
		nthreads = n_threads;
	}
	void* operator new(size_t i) {
		return _mm_malloc(i, 64);
	}
	void operator delete(void* p) {
		_mm_free(p);
	}
};



static size_t select_work_size_blake(void)
{
	size_t              work_size =
		64 * /* thread per wavefront */
		BLAKE_WPS * /* wavefront per simd */
		4 * /* simd per compute unit */
		36;
	// Make the work group size a multiple of the nr of wavefronts, while
	// dividing the number of inputs. This results in the worksize being a
	// power of 2.
	while (NR_INPUTS % work_size)
		work_size += 64;
	//debug("Blake: work size %zd\n", work_size);
	return work_size;
}

static void sort_pair(uint32_t *a, uint32_t len)
{
	uint32_t    *b = a + len;
	uint32_t     tmp, need_sorting = 0;
	for (uint32_t i = 0; i < len; i++)
		if (need_sorting || a[i] > b[i])
		{
			need_sorting = 1;
			tmp = a[i];
			a[i] = b[i];
			b[i] = tmp;
		}
		else if (a[i] < b[i])
			return;
}

static uint32_t verify_sol(sols_t *sols, unsigned sol_i)
{
	uint32_t  *inputs = sols->values[sol_i];
	uint32_t  seen_len = (1 << (PREFIX + 1)) / 8;
	uint8_t seen[(1 << (PREFIX + 1)) / 8];
	uint32_t  i;
	uint8_t tmp;
	// look for duplicate inputs
	memset(seen, 0, seen_len);
	for (i = 0; i < (1 << PARAM_K); i++)
	{
		tmp = seen[inputs[i] / 8];
		seen[inputs[i] / 8] |= 1 << (inputs[i] & 7);
		if (tmp == seen[inputs[i] / 8])
		{
			// at least one input value is a duplicate
			sols->valid[sol_i] = 0;
			return 0;
		}
	}
	// the valid flag is already set by the GPU, but set it again because
	// I plan to change the GPU code to not set it
	sols->valid[sol_i] = 1;
	// sort the pairs in place
	for (uint32_t level = 0; level < PARAM_K; level++)
		for (i = 0; i < (1 << PARAM_K); i += (2 << level))
			sort_pair(&inputs[i], 1 << level);
	return 1;
}

static void compress(uint8_t *out, uint32_t *inputs, uint32_t n)
{
	uint32_t byte_pos = 0;
	int32_t bits_left = PREFIX + 1;
	uint8_t x = 0;
	uint8_t x_bits_used = 0;
	uint8_t *pOut = out;
	while (byte_pos < n)
	{
		if (bits_left >= 8 - x_bits_used)
		{
			x |= inputs[byte_pos] >> (bits_left - 8 + x_bits_used);
			bits_left -= 8 - x_bits_used;
			x_bits_used = 8;
		}
		else if (bits_left > 0)
		{
			uint32_t mask = ~(-1 << (8 - x_bits_used));
			mask = ((~mask) >> bits_left) & mask;
			x |= (inputs[byte_pos] << (8 - x_bits_used - bits_left)) & mask;
			x_bits_used += bits_left;
			bits_left = 0;
		}
		else if (bits_left <= 0)
		{
			assert(!bits_left);
			byte_pos++;
			bits_left = PREFIX + 1;
		}
		if (x_bits_used == 8)
		{
			*pOut++ = x;
			x = x_bits_used = 0;
		}
	}
}

sa_cuda_context::sa_cuda_context(int tpb, int blocks, int id)
	: threadsperblock(tpb), totalblocks(blocks), device_id(id)
{
	checkCudaErrors(cudaSetDevice(device_id));
	checkCudaErrors(cudaDeviceReset());
	checkCudaErrors(cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync));
	checkCudaErrors(cudaDeviceSetCacheConfig(cudaFuncCachePreferL1));

	eq = new c_context(threadsperblock * totalblocks);
#ifdef ENABLE_DEBUG
	size_t              dbg_size = NR_ROWS;
#else
	size_t              dbg_size = 1;
#endif

	checkCudaErrors(cudaMalloc((void**)&eq->buf_dbg, dbg_size));
	checkCudaErrors(cudaMalloc((void**)&eq->buf_ht[0], HT_SIZE));
	checkCudaErrors(cudaMalloc((void**)&eq->buf_ht[1], HT_SIZE));
	checkCudaErrors(cudaMalloc((void**)&eq->buf_sols, sizeof(sols_t)));
	checkCudaErrors(cudaMalloc((void**)&eq->rowCounters[0], NR_ROWS));
        checkCudaErrors(cudaMalloc((void**)&eq->rowCounters[1], NR_ROWS));
	checkCudaErrors(cudaMallocHost(&(eq->sols), sizeof(*eq->sols)));
}

sa_cuda_context::~sa_cuda_context()
{
	checkCudaErrors(cudaSetDevice(device_id));
	checkCudaErrors(cudaFreeHost(eq->sols));
	checkCudaErrors(cudaDeviceReset());
	delete eq;
}

void sa_cuda_context::solve(const char * tequihash_header, unsigned int tequihash_header_len, const char * nonce, unsigned int nonce_len, std::function<bool()> cancelf, std::function<void(const std::vector<uint32_t>&, size_t, const unsigned char*)> solutionf, std::function<void(void)> hashdonef)
{
	checkCudaErrors(cudaSetDevice(device_id));

	unsigned char context[140];
	memset(context, 0, 140);
	memcpy(context, tequihash_header, tequihash_header_len);
	memcpy(context + tequihash_header_len, nonce, nonce_len);
	//printf("NR_SLOTS=%d NR_ROWS=%d\n",NR_SLOTS,NR_ROWS);
	c_context *miner = eq;
	
	//FUNCTION<<<totalblocks, threadsperblock>>>(ARGUMENTS)

	blake2b_state_t initialCtx;
	zcash_blake2b_init(&initialCtx, ZCASH_HASH_LEN, PARAM_N, PARAM_K);
	zcash_blake2b_update(&initialCtx, (const uint8_t*)context, 128, 0);

	void* buf_blake_st;
	checkCudaErrors(cudaMalloc((void**)&buf_blake_st, sizeof(blake2b_state_s)));
	checkCudaErrors(cudaMemcpy(buf_blake_st, &initialCtx, sizeof(blake2b_state_s), cudaMemcpyHostToDevice));
	
	for (unsigned round = 0; round < PARAM_K; round++) {
//		if (round < 2) {
			//every round
			checkCudaErrors(cudaMemset(miner->rowCounters[round % 2],0,NR_ROWS));
//			kernel_init_ht<<<NR_ROWS / ROWS_PER_UINT / 256, 256>>>(miner->buf_ht[round & 1],(uint*)miner->rowCounters[round % 2]);
//			printf("%d %d %d\n",NR_ROWS / ROWS_PER_UINT / 256, NR_ROWS,ROWS_PER_UINT);
//			exit(-1);
//		}
		if (!round)	{
			miner->global_ws = select_work_size_blake();
		} else {
			miner->global_ws = NR_ROWS;
		}
		// cancel function
		switch (round) {
		case 0:
			kernel_round0<<<NR_INPUTS/THRD,THRD>>>((ulong*)buf_blake_st, miner->buf_ht[round & 1], (uint*)miner->rowCounters[round % 2],(uint*)miner->buf_dbg);
//			printf("%d %d\n",totalblocks, NR_INPUTS);
//                        exit(-1);
			break;
		case 1:
			kernel_round1<<<BLK,THRD>>>(miner->buf_ht[(round - 1) & 1], miner->buf_ht[round & 1],(uint*)miner->rowCounters[(round - 1) % 2],(uint*)miner->rowCounters[round % 2], (uint*)miner->buf_dbg);
			//exit(0);
			break;
		case 2:
			kernel_round2<<<BLK,THRD>>>(miner->buf_ht[(round - 1) & 1], miner->buf_ht[round & 1], (uint*)miner->rowCounters[(round - 1) % 2],(uint*)miner->rowCounters[round % 2], (uint*)miner->buf_dbg);
//			exit(0);
			break;
		case 3:
			kernel_round3<<<BLK,THRD>>>(miner->buf_ht[(round - 1) & 1], miner->buf_ht[round & 1],(uint*)miner->rowCounters[(round - 1) % 2],(uint*)miner->rowCounters[round % 2], (uint*)miner->buf_dbg);
			break;
		case 4:
			kernel_round4<<<BLK,THRD>>>(miner->buf_ht[(round - 1) & 1], miner->buf_ht[round & 1],(uint*)miner->rowCounters[(round - 1) % 2],(uint*)miner->rowCounters[round % 2], (uint*)miner->buf_dbg);
			break;
		case 5:
			kernel_round5<<<BLK,THRD>>>(miner->buf_ht[(round - 1) & 1], miner->buf_ht[round & 1],(uint*)miner->rowCounters[(round - 1) % 2],(uint*)miner->rowCounters[round % 2], (uint*)miner->buf_dbg);
//exit(0);
			break;
		case 6:
			kernel_round6<<<BLK,THRD>>>(miner->buf_ht[(round - 1) & 1], miner->buf_ht[round & 1],(uint*)miner->rowCounters[(round - 1) % 2],(uint*)miner->rowCounters[round % 2], (uint*)miner->buf_dbg);
	//	exit(0);
			break;
		case 7:
			kernel_round7<<<BLK,THRD>>>(miner->buf_ht[(round - 1) & 1], miner->buf_ht[round & 1],(uint*)miner->rowCounters[(round - 1) % 2],(uint*)miner->rowCounters[round % 2], (uint*)miner->buf_dbg);
//			exit(0);
			break;
		case 8:
			kernel_round8<<<BLK,THRD>>>(miner->buf_ht[(round - 1) & 1], miner->buf_ht[round & 1],(uint*)miner->rowCounters[(round - 1) % 2],(uint*)miner->rowCounters[round % 2], (uint*)miner->buf_dbg, (sols_t*)miner->buf_sols);
			break;
		}
		if (cancelf()) return;
	}
	kernel_sols<<<NR_ROWS/32,32>>>(miner->buf_ht[0], miner->buf_ht[1], (sols_t*)miner->buf_sols,(uint*)miner->rowCounters[0],(uint*)miner->rowCounters[1]);

	checkCudaErrors(cudaMemcpy(miner->sols, miner->buf_sols, sizeof(*miner->sols), cudaMemcpyDeviceToHost));
	checkCudaErrors(cudaFree(buf_blake_st));

	if (miner->sols->nr > MAX_SOLS)
		miner->sols->nr = MAX_SOLS;

	for (unsigned sol_i = 0; sol_i < miner->sols->nr; sol_i++) {
		verify_sol(miner->sols, sol_i);
	}

	uint8_t proof[COMPRESSED_PROOFSIZE * 2];
	for (uint32_t i = 0; i < miner->sols->nr; i++) {
		if (miner->sols->valid[i]) {
			compress(proof, (uint32_t *)(miner->sols->values[i]), 1 << PARAM_K);
			solutionf(std::vector<uint32_t>(0), 1344, proof);
		}
	}
	hashdonef();
}
