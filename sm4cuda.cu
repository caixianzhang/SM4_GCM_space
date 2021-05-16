#include <string.h>
#include <stdio.h>
#include <time.h>
#include <stdint.h>
#include <stdlib.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "sm4cuda.cuh"

//S盒参数
uint8_t SboxTable[256] = { \
	0xd6,0x90,0xe9,0xfe,0xcc,0xe1,0x3d,0xb7,0x16,0xb6,0x14,0xc2,0x28,0xfb,0x2c,0x05, \
	0x2b,0x67,0x9a,0x76,0x2a,0xbe,0x04,0xc3,0xaa,0x44,0x13,0x26,0x49,0x86,0x06,0x99, \
	0x9c,0x42,0x50,0xf4,0x91,0xef,0x98,0x7a,0x33,0x54,0x0b,0x43,0xed,0xcf,0xac,0x62, \
	0xe4,0xb3,0x1c,0xa9,0xc9,0x08,0xe8,0x95,0x80,0xdf,0x94,0xfa,0x75,0x8f,0x3f,0xa6, \
	0x47,0x07,0xa7,0xfc,0xf3,0x73,0x17,0xba,0x83,0x59,0x3c,0x19,0xe6,0x85,0x4f,0xa8, \
	0x68,0x6b,0x81,0xb2,0x71,0x64,0xda,0x8b,0xf8,0xeb,0x0f,0x4b,0x70,0x56,0x9d,0x35, \
	0x1e,0x24,0x0e,0x5e,0x63,0x58,0xd1,0xa2,0x25,0x22,0x7c,0x3b,0x01,0x21,0x78,0x87, \
	0xd4,0x00,0x46,0x57,0x9f,0xd3,0x27,0x52,0x4c,0x36,0x02,0xe7,0xa0,0xc4,0xc8,0x9e, \
	0xea,0xbf,0x8a,0xd2,0x40,0xc7,0x38,0xb5,0xa3,0xf7,0xf2,0xce,0xf9,0x61,0x15,0xa1, \
	0xe0,0xae,0x5d,0xa4,0x9b,0x34,0x1a,0x55,0xad,0x93,0x32,0x30,0xf5,0x8c,0xb1,0xe3, \
	0x1d,0xf6,0xe2,0x2e,0x82,0x66,0xca,0x60,0xc0,0x29,0x23,0xab,0x0d,0x53,0x4e,0x6f, \
	0xd5,0xdb,0x37,0x45,0xde,0xfd,0x8e,0x2f,0x03,0xff,0x6a,0x72,0x6d,0x6c,0x5b,0x51, \
	0x8d,0x1b,0xaf,0x92,0xbb,0xdd,0xbc,0x7f,0x11,0xd9,0x5c,0x41,0x1f,0x10,0x5a,0xd8, \
	0x0a,0xc1,0x31,0x88,0xa5,0xcd,0x7b,0xbd,0x2d,0x74,0xd0,0x12,0xb8,0xe5,0xb4,0xb0, \
	0x89,0x69,0x97,0x4a,0x0c,0x96,0x77,0x7e,0x65,0xb9,0xf1,0x09,0xc5,0x6e,0xc6,0x84, \
	0x18,0xf0,0x7d,0xec,0x3a,0xdc,0x4d,0x20,0x79,0xee,0x5f,0x3e,0xd7,0xcb,0x39,0x48, \
};

/* System parameter */
uint32_t FK[4] = { 0xa3b1bac6,0x56aa3350,0x677d9197,0xb27022dc };

/* fixed parameter */
uint32_t CK[32] = { \
	0x00070e15,0x1c232a31,0x383f464d,0x545b6269, \
	0x70777e85,0x8c939aa1,0xa8afb6bd,0xc4cbd2d9, \
	0xe0e7eef5,0xfc030a11,0x181f262d,0x343b4249, \
	0x50575e65,0x6c737a81,0x888f969d,0xa4abb2b9, \
	0xc0c7ced5,0xdce3eaf1,0xf8ff060d,0x141b2229, \
	0x30373e45,0x4c535a61,0x686f767d,0x848b9299, \
	0xa0a7aeb5,0xbcc3cad1,0xd8dfe6ed,0xf4fb0209, \
	0x10171e25,0x2c333a41,0x484f565d,0x646b7279, \
};

/*
   行移位函数 C++版本
   b:需要移动的数组指针
   i:需要移动的位数
   n:返回值，
 */
inline void GET_UINT_BE(uint32_t *n, uint8_t *b, uint32_t i)
{
	(*n) = (((uint32_t)b[i]) << 24) | (((uint32_t)b[i + 1]) << 16) | (((uint32_t)b[i + 2]) << 8) | (uint32_t)b[i + 3];
}

/*
	行移位函数 C++版本逆运算
	b:需要移动的数组指针
	i:需要移动的位数
	n:输入值，
*/
inline void PUT_UINT_BE(uint32_t n, uint8_t *b, uint32_t i)
{
	//取n的高四位
	b[i + 0] = (uint8_t)(n >> 24);

	//取n的次高四位
	b[i + 1] = (uint8_t)(n >> 16);

	//取n的次低四位
	b[i + 2] = (uint8_t)(n >> 8);

	//取n的低四位
	b[i + 3] = (uint8_t)n;
}

/*
	S盒替换
*/
inline uint8_t sm4Sbox(uint8_t inch)
{
	return SboxTable[inch];
}

/*
	循环左移函数，即将x循环左移n位
*/
inline uint32_t ROTL(uint32_t x, uint32_t n)
{
	return (x << n) | (x >> (32 - n));
}

/*
	互换a b的值
*/
inline void SWAP(uint32_t *a, uint32_t *b)
{
	uint32_t c = *a;
	*a = *b;
	*b = c;
}

uint32_t sm4Lt(uint32_t ka)
{
	uint8_t a[4];
	PUT_UINT_BE(ka, a, 0);

	//查表替换
	a[0] = sm4Sbox(a[0]);
	a[1] = sm4Sbox(a[1]);
	a[2] = sm4Sbox(a[2]);
	a[3] = sm4Sbox(a[3]);

	//将查表后的数放到bb数组中去
	uint32_t bb = 0;
	GET_UINT_BE(&bb, a, 0);

	//bb分别与其循环左移2位，10位，18位，24位数相异或， 得到的值返回
	return bb ^ (ROTL(bb, 2)) ^ (ROTL(bb, 10)) ^ (ROTL(bb, 18)) ^ (ROTL(bb, 24));
}

uint32_t sm4F(uint32_t x0, uint32_t x1, uint32_t x2, uint32_t x3, uint32_t rk)
{
	return (x0^sm4Lt(x1^x2^x3^rk));
}


/*
	密钥拓展函数
*/
uint32_t sm4CalciRK(uint32_t ka)
{
	uint8_t a[4];
	PUT_UINT_BE(ka, a, 0);
	a[0] = sm4Sbox(a[0]);
	a[1] = sm4Sbox(a[1]);
	a[2] = sm4Sbox(a[2]);
	a[3] = sm4Sbox(a[3]);

	uint32_t bb = 0;
	GET_UINT_BE(&bb, a, 0);
	return bb ^ (ROTL(bb, 13)) ^ (ROTL(bb, 23));
}

/*
	SK:值结果参数，用于填写扩展密钥
	key:初始密钥(128bit)
*/
void sm4_setkey(uint32_t SK[32], uint8_t key[16])
{
	uint32_t MK[4];
	GET_UINT_BE(&MK[0], key, 0);
	GET_UINT_BE(&MK[1], key, 4);
	GET_UINT_BE(&MK[2], key, 8);
	GET_UINT_BE(&MK[3], key, 12);

	//初始轮密钥
	uint32_t k[36];
	k[0] = MK[0] ^ FK[0];
	k[1] = MK[1] ^ FK[1];
	k[2] = MK[2] ^ FK[2];
	k[3] = MK[3] ^ FK[3];

	for (int i = 0; i < 32; i++)
	{
		k[i + 4] = k[i] ^ (sm4CalciRK(k[i + 1] ^ k[i + 2] ^ k[i + 3] ^ CK[i]));
		SK[i] = k[i + 4];
	}
}

/*
	SM4轮函数
*/
void sm4_one_round(uint32_t sk[32], uint8_t input[16], uint8_t output[16])
{

	uint32_t ulbuf[36];
	memset(ulbuf, 0, sizeof(ulbuf));

	GET_UINT_BE(&ulbuf[0], input, 0);
	GET_UINT_BE(&ulbuf[1], input, 4);
	GET_UINT_BE(&ulbuf[2], input, 8);
	GET_UINT_BE(&ulbuf[3], input, 12);

	for (int i = 0; i < 32; i++)
	{
		ulbuf[i + 4] = sm4F(ulbuf[i], ulbuf[i + 1], ulbuf[i + 2], ulbuf[i + 3], sk[i]);
	}

	PUT_UINT_BE(ulbuf[35], output, 0);
	PUT_UINT_BE(ulbuf[34], output, 4);
	PUT_UINT_BE(ulbuf[33], output, 8);
	PUT_UINT_BE(ulbuf[32], output, 12);
}

/*
	加密模式密钥拓展
	ctx：值结果参数，函数执行完毕后会填写加密密钥相关信息，
	key: 加密密钥（长度128bit）
*/
void sm4_setkey_enc(sm4_context *ctx, uint8_t key[16])
{
	ctx->mode = SM4_ENCRYPT;
	sm4_setkey(ctx->sk, key);
}

/*
	解密模式密钥拓展
	ctx：值结果参数，函数执行完毕后会填写加密密钥相关信息，
	key: 加密密钥（长度128bit）
*/
void sm4_setkey_dec(sm4_context *ctx, uint8_t key[16])
{
	ctx->mode = SM4_DECRYPT;
	sm4_setkey(ctx->sk, key);
	for (int i = 0; i < 16; i++)
	{
		SWAP(&(ctx->sk[i]), &(ctx->sk[31 - i]));
	}
}

/*
 * SM4-ECB block encryption/decryption
 *
 * SM4-ECB模式加解密函数
 * ctx：值结果参数，子密钥参数指针
 * mode:加解密模式，SM4不区分加解密模式，密文进则明文出，明文进则密文出
 * input:数据输入(16字节)
 * output:数据输出(16字节)
 */
void sm4_crypt_ecb(sm4_context *ctx, int length, uint8_t *input, uint8_t *output)
{
	while (length > 0)
	{
		sm4_one_round(ctx->sk, input, output);
		input += 16;
		output += 16;
		length -= 16;
	}
}


/*
	线程束32个线程的读写地址定位查找表
*/
uint32_t smem_offset[32] = {
	0 * 132 + 0 * 16, 0 * 132 + 1 * 16, 0 * 132 + 2 * 16, 0 * 132 + 3 * 16, 0 * 132 + 4 * 16, 0 * 132 + 5 * 16, 0 * 132 + 6 * 16, 0 * 132 + 7 * 16, \
	1 * 132 + 0 * 16, 1 * 132 + 1 * 16, 1 * 132 + 2 * 16, 1 * 132 + 3 * 16, 1 * 132 + 4 * 16, 1 * 132 + 5 * 16, 1 * 132 + 6 * 16, 1 * 132 + 7 * 16, \
	2 * 132 + 0 * 16, 2 * 132 + 1 * 16, 2 * 132 + 2 * 16, 2 * 132 + 3 * 16, 2 * 132 + 4 * 16, 2 * 132 + 5 * 16, 2 * 132 + 6 * 16, 2 * 132 + 7 * 16, \
	3 * 132 + 0 * 16, 3 * 132 + 1 * 16, 3 * 132 + 2 * 16, 3 * 132 + 3 * 16, 3 * 132 + 4 * 16, 3 * 132 + 5 * 16, 3 * 132 + 6 * 16, 3 * 132 + 7 * 16, \
};

//有限域乘法查找表
uint8_t gfmult_table[16][256][16];
uint8_t ency0[16];

//每个线程块共享SK,ency0,lenAC
__constant__ uint32_t constant_sk[32];
__constant__ uint8_t  constant_ency0[16];
__constant__ uint8_t  constant_lenAC[16];

void otherT(uint8_t T[16][256][16])
{
	int i = 0, j = 0, k = 0;
	uint64_t vh, vl;
	uint64_t zh, zl;
	for (i = 0; i < 256; i++)
	{
		vh = ((uint64_t)T[0][i][0] << 56) ^ ((uint64_t)T[0][i][1] << 48) ^ \
			((uint64_t)T[0][i][2] << 40) ^ ((uint64_t)T[0][i][3] << 32) ^ \
			((uint64_t)T[0][i][4] << 24) ^ ((uint64_t)T[0][i][5] << 16) ^ \
			((uint64_t)T[0][i][6] << 8) ^ ((uint64_t)T[0][i][7]);

		vl = ((uint64_t)T[0][i][8] << 56) ^ ((uint64_t)T[0][i][9] << 48) ^ \
			((uint64_t)T[0][i][10] << 40) ^ ((uint64_t)T[0][i][11] << 32) ^ \
			((uint64_t)T[0][i][12] << 24) ^ ((uint64_t)T[0][i][13] << 16) ^ \
			((uint64_t)T[0][i][14] << 8) ^ ((uint64_t)T[0][i][15]);

		zh = zl = 0;

		for (j = 0; j <= 120; j++)
		{
			if ((j > 0) && (0 == j % 8))
			{
				zh ^= vh;
				zl ^= vl;
				for (k = 1; k <= 16 / 2; k++)
				{
					T[j / 8][i][16 / 2 - k] = (uint8_t)zh;
					zh = zh >> 8;
					T[j / 8][i][16 - k] = (uint8_t)zl;
					zl = zl >> 8;
				}
				zh = zl = 0;
			}
			if (vl & 0x1)
			{
				vl = vl >> 1;
				if (vh & 0x1) { vl ^= 0x8000000000000000; }
				vh = vh >> 1;
				vh ^= 0xe100000000000000;
			}
			else
			{
				vl = vl >> 1;
				if (vh & 0x1) { vl ^= 0x8000000000000000; }
				vh = vh >> 1;
			}
		}
	}
}

//生成GF乘法表
void computeTable(uint8_t T[16][256][16], uint8_t H[16])
{
	// zh is the higher 64-bit, zl is the lower 64-bit
	uint64_t zh = 0, zl = 0;
	// vh is the higher 64-bit, vl is the lower 64-bit
	uint64_t vh = ((uint64_t)H[0] << 56) ^ ((uint64_t)H[1] << 48) ^ \
		((uint64_t)H[2] << 40) ^ ((uint64_t)H[3] << 32) ^ \
		((uint64_t)H[4] << 24) ^ ((uint64_t)H[5] << 16) ^ \
		((uint64_t)H[6] << 8) ^ ((uint64_t)H[7]);

	uint64_t vl = ((uint64_t)H[8] << 56) ^ ((uint64_t)H[9] << 48) ^ \
		((uint64_t)H[10] << 40) ^ ((uint64_t)H[11] << 32) ^ \
		((uint64_t)H[12] << 24) ^ ((uint64_t)H[13] << 16) ^ \
		((uint64_t)H[14] << 8) ^ ((uint64_t)H[15]);

	uint8_t temph;

	uint64_t tempvh = vh;
	uint64_t tempvl = vl;
	int i = 0, j = 0;
	for (i = 0; i < 256; i++)
	{
		temph = (uint8_t)i;
		vh = tempvh;
		vl = tempvl;
		zh = zl = 0;

		for (j = 0; j < 8; j++)
		{
			if (0x80 & temph)
			{
				zh ^= vh;
				zl ^= vl;
			}
			if (vl & 0x1)
			{
				vl = vl >> 1;
				if (vh & 0x1) { vl ^= 0x8000000000000000; }
				vh = vh >> 1;
				vh ^= 0xe100000000000000;
			}
			else
			{
				vl = vl >> 1;
				if (vh & 0x1) { vl ^= 0x8000000000000000; }
				vh = vh >> 1;
			}
			temph = temph << 1;
		}
		// get result
		for (j = 1; j <= 16 / 2; j++)
		{
			T[0][i][16 / 2 - j] = (uint8_t)zh;
			zh = zh >> 8;
			T[0][i][16 - j] = (uint8_t)zl;
			zl = zl >> 8;
		}
	}
	otherT(T);
}

/**
 * return the value of (output.H) by looking up tables
 */
void multi(uint8_t T[16][256][16], uint8_t *output)
{
	uint8_t i, j;
	uint8_t temp[16];
	for (i = 0; i < 16; i++)
	{
		temp[i] = output[i];
		output[i] = 0;
	}
	for (i = 0; i < 16; i++)
	{
		for (j = 0; j < 16; j++)
		{
			output[j] ^= T[i][*(temp + i)][j];
		}
	}
}

/*
 * a: additional authenticated data
 * c: the cipher text or initial vector
 */
void ghash(uint8_t T[16][256][16], uint8_t *add, size_t add_len, uint8_t *cipher, size_t length, uint8_t *output)
{
	/* x0 = 0 */
	*(uint64_t *)output = 0;
	*((uint64_t *)output + 1) = 0;

	/* compute with add */
	int i = 0;
	for (i = 0; i < add_len / 16; i++)
	{
		*(uint64_t *)output ^= *(uint64_t *)add;
		*((uint64_t *)output + 1) ^= *((uint64_t *)add + 1);
		add += 16;
		multi(T, output);
	}

	if (add_len % 16)
	{
		// the remaining add
		for (i = 0; i < add_len % 16; i++)
		{
			*(output + i) ^= *(add + i);
		}
		multi(T, output);
	}

	/* compute with cipher text */
	for (i = 0; i < length / 16; i++)
	{
		*(uint64_t *)output ^= *(uint64_t *)cipher;
		*((uint64_t *)output + 1) ^= *((uint64_t *)cipher + 1);
		cipher += 16;
		multi(T, output);
	}
	if (length % 16)
	{
		// the remaining cipher
		for (i = 0; i < length % 16; i++)
		{
			*(output + i) ^= *(cipher + i);
		}
		multi(T, output);
	}

	/* eor (len(A)||len(C)) */
	uint64_t temp_len = (uint64_t)(add_len * 8); // len(A) = (uint64_t)(add_len*8)
	for (i = 1; i <= 16 / 2; i++)
	{
		output[16 / 2 - i] ^= (uint8_t)temp_len;
		temp_len = temp_len >> 8;
	}
	temp_len = (uint64_t)(length * 8); // len(C) = (uint64_t)(length*8)
	for (i = 1; i <= 16 / 2; i++)
	{
		output[16 - i] ^= (uint8_t)temp_len;
		temp_len = temp_len >> 8;
	}
	multi(T, output);
}

/*
**	加密算法核函数，完成SM4-CTR模式加密，每个线程加密一个序号，之后与明文数据块异或，生成密文
**	dev_SboxTable:S盒
**	counter:数据块序号
**	dev_input:明文数据输入
**	dev_output:密文数据输出
*/
__global__ void kernal_enc(uint8_t *const __restrict__ dev_SboxTable, \
	uint32_t *const __restrict__ dev_smem_offset, \
	uint32_t counter, \
	uint8_t dev_iv[PARTICLE_SIZE / STREAM_SIZE], \
	uint8_t dev_input[PARTICLE_SIZE / STREAM_SIZE], \
	uint8_t dev_output[PARTICLE_SIZE / STREAM_SIZE])
{
	__shared__ uint8_t smem[(32 * 4 + 4) * 4 * (BLOCK_SIZE / 32)];
	//确定对齐合并访存时，线程束中的每个线程在全局内存上的读写地址偏移量
	uint32_t dev_offset = blockIdx.x * blockDim.x * 16 + threadIdx.x * 4;
	//确定对齐合并访存时，线程束中的每个线程在共享内存上的读写地址偏移量
	uint32_t share_offset = (threadIdx.x / 32) * (32 * 4 + 4) + (threadIdx.x % 32) * 4;

	//以对齐合并访存的模式从全局内存读取IV
	{
		uint8_t *read = dev_iv + dev_offset;
		uint8_t *write = smem + share_offset;
		#pragma unroll 4
		for (int i = 0; i < 4; i++)
		{
			*(uint32_t *)(write + i * ((BLOCK_SIZE / 32) * (32 * 4 + 4))) = *(uint32_t *)(read + i * BLOCK_SIZE * 4);
		}
	}
	
	//确定线程束中，各个线程的共享内存读写地址
	uint8_t *smem_rw_pos = smem + (threadIdx.x / 32) * ((32 * 4 + 4) * 4) + dev_smem_offset[threadIdx.x % 32];

	__syncthreads();

	{
		uint32_t ulbuf[5];

		{
			//各个线程读取各自IV
			uint8_t tidCTR[16];
			*(uint32_t *)(tidCTR + 0 * 4) = *(uint32_t *)(smem_rw_pos + 0 * 4);
			*(uint32_t *)(tidCTR + 1 * 4) = *(uint32_t *)(smem_rw_pos + 1 * 4);
			*(uint32_t *)(tidCTR + 2 * 4) = *(uint32_t *)(smem_rw_pos + 2 * 4);
			*(uint32_t *)(tidCTR + 3 * 4) = counter;

			#pragma unroll 4
			for (int i = 0; i < 4; i++)
			{
				ulbuf[i] = (((uint32_t)tidCTR[i * 4]) << 24) | \
					(((uint32_t)tidCTR[i * 4 + 1]) << 16) | \
					(((uint32_t)tidCTR[i * 4 + 2]) << 8) | \
					(uint32_t)tidCTR[i * 4 + 3];
			}
		}

		//32轮迭代运算
		{
			uint32_t temp;
			uint8_t a[4];
			uint32_t bb;

			#pragma unroll 32
			for (int i = 0; i < 32; i++)
			{
				temp = ulbuf[(i + 1) % 5] ^ ulbuf[(i + 2) % 5] ^ ulbuf[(i + 3) % 5] ^ constant_sk[i];
				a[0] = (uint8_t)(temp >> 24);
				a[1] = (uint8_t)(temp >> 16);
				a[2] = (uint8_t)(temp >> 8);
				a[3] = (uint8_t)temp;
				a[0] = dev_SboxTable[a[0]];
				a[1] = dev_SboxTable[a[1]];
				a[2] = dev_SboxTable[a[2]];
				a[3] = dev_SboxTable[a[3]];
				bb = (((uint32_t)a[0]) << 24) | (((uint32_t)a[1]) << 16) | (((uint32_t)a[2]) << 8) | (uint32_t)a[3];
				bb = bb ^ ((bb << 2) | (bb >> 30)) ^ ((bb << 10) | (bb >> 22)) ^ ((bb << 18) | (bb >> 14)) ^ ((bb << 24) | (bb >> 8));
				ulbuf[(i + 4) % 5] = ulbuf[(i + 0) % 5] ^ bb;
			}
		}

		{
			//填写本线程密文输出起始地址(矩形存储模式)，密文存放在共享内存
			uint8_t temp[4];
			temp[0] = (uint8_t)(ulbuf[0] >> 24);
			temp[1] = (uint8_t)(ulbuf[0] >> 16);
			temp[2] = (uint8_t)(ulbuf[0] >> 8);
			temp[3] = (uint8_t)ulbuf[0];
			*(uint32_t *)(smem_rw_pos + 0 * 4) = *(uint32_t *)temp;

			temp[0] = (uint8_t)(ulbuf[4] >> 24);
			temp[1] = (uint8_t)(ulbuf[4] >> 16);
			temp[2] = (uint8_t)(ulbuf[4] >> 8);
			temp[3] = (uint8_t)ulbuf[4];
			*(uint32_t *)(smem_rw_pos + 1 * 4) = *(uint32_t *)temp;

			temp[0] = (uint8_t)(ulbuf[3] >> 24);
			temp[1] = (uint8_t)(ulbuf[3] >> 16);
			temp[2] = (uint8_t)(ulbuf[3] >> 8);
			temp[3] = (uint8_t)ulbuf[3];
			*(uint32_t *)(smem_rw_pos + 2 * 4) = *(uint32_t *)temp;

			temp[0] = (uint8_t)(ulbuf[2] >> 24);
			temp[1] = (uint8_t)(ulbuf[2] >> 16);
			temp[2] = (uint8_t)(ulbuf[2] >> 8);
			temp[3] = (uint8_t)ulbuf[2];
			*(uint32_t *)(smem_rw_pos + 3 * 4) = *(uint32_t *)temp;
		}
	}

	__syncthreads();

	//以对齐合并访存的方式读取明文，将加密后的序号与明文异或后生成密文，生成密文后再以对齐合并访存的方式写回全局内存
	{
		uint8_t *read = dev_input + dev_offset;
		uint8_t *write = dev_output + dev_offset;
		uint8_t *cipher = smem + share_offset;

		#pragma unroll 4
		for (int i = 0; i < 4; i++)
		{
			*(uint32_t *)(write + i * BLOCK_SIZE * 4) = (*(uint32_t *)(cipher + i * ((BLOCK_SIZE / 32) * (32 * 4 + 4)))) ^ (*(uint32_t *)(read + i * BLOCK_SIZE * 4));
		}
	}

}
/*
**	有限域乘法加法运算核函数
**	dev_gfmult_table:有限域乘法表
**	dev_cipher:密文输入(矩形存储模式)
**	dev_gfmult:有限域乘法结果(矩形存储模式)
*/
__global__ void kernal_gfmult(uint32_t *const __restrict__ dev_smem_offset, \
	uint8_t dev_gfmult_table[16][256][16], \
	uint8_t dev_cipher[PARTICLE_SIZE / STREAM_SIZE], \
	uint8_t dev_gfmult[PARTICLE_SIZE / STREAM_SIZE])
{
	__shared__ uint8_t smem[(32 * 4 + 4) * 4 * (BLOCK_SIZE / 32)];

	uint32_t dev_offset = blockIdx.x * blockDim.x * 16 + threadIdx.x * 4;
	uint32_t share_offset = (threadIdx.x / 32) * (32 * 4 + 4) + (threadIdx.x % 32) * 4;

	//以对齐合并访存的模式进行有限域加法
	{
		uint8_t *read_cipher = dev_cipher + dev_offset;
		uint8_t *read_gfmult = dev_gfmult + dev_offset;
		uint8_t *write = smem + share_offset;

		#pragma unroll 4
		for (int i = 0; i < 4; i++)
		{
			*(uint32_t *)(write + i * ((BLOCK_SIZE / 32) * (32 * 4 + 4))) = (*(uint32_t *)(read_cipher + i * BLOCK_SIZE * 4)) ^ (*(uint32_t *)(read_gfmult + i * BLOCK_SIZE * 4));
		}
	}

	//确定线程束中，各个线程的共享内存读写地址
	uint8_t *smem_rw_pos = smem + (threadIdx.x / 32) * ((32 * 4 + 4) * 4) + dev_smem_offset[threadIdx.x % 32];

	__syncthreads();

	//有限域乘法
	{
		uint8_t temp;
		uint8_t read[4];

		//暂存GF乘法结果
		uint8_t tid_gfmult[16];

		#pragma unroll 4
		for (int i = 0; i < 4; i++)
		{
			*(uint32_t *)(tid_gfmult + i * 4) = 0;
		}

		#pragma unroll 4
		for (int i = 0; i < 4; i++)
		{
			*(uint32_t *)read = *(uint32_t *)(smem_rw_pos + i * 4);

			#pragma unroll 4
			for (int j = 0; j < 4; j++)
			{
				temp = read[j];
				#pragma unroll 16
				for (int k = 0; k < 16; k++)
				{
					tid_gfmult[k] ^= dev_gfmult_table[i * 4 + j][temp][k];
				}
			}
		}

		//将本数据块的有限域乘法的结果写回共享内存
		#pragma unroll 4
		for (int i = 0; i < 4; i++)
		{
			*(uint32_t *)(smem_rw_pos + i * 4) = *(uint32_t *)(tid_gfmult + i * 4);
		}
	}

	__syncthreads();

	//以对齐合并访存的方式将共享内存中的乘法结果写回全局内存。
	{
		uint8_t *write = dev_gfmult + dev_offset;
		uint8_t *read = smem + share_offset;

		#pragma unroll 4
		for (int i = 0; i < 4; i++)
		{
			*(uint32_t *)(write + i * BLOCK_SIZE * 4) = *(uint32_t *)(read + i * ((BLOCK_SIZE / 32) * (32 * 4 + 4)));
		}
	}
}

/*
**	本核函数完成计算每个线程最终的GHASH结果
**	dev_gfmult_table;有限域乘法表
**	dev_gfmult:有限域乘法结果
*/
__global__ void kernal_final(uint8_t dev_gfmult_table[16][256][16], \
	uint32_t *const __restrict__ dev_smem_offset, \
	uint8_t dev_gfmult[PARTICLE_SIZE / STREAM_SIZE])
{
	__shared__ uint8_t smem[(32 * 4 + 4) * 4 * (BLOCK_SIZE / 32)];

	uint32_t dev_offset = blockIdx.x * blockDim.x * 16 + threadIdx.x * 4;
	uint32_t share_offset = (threadIdx.x / 32) * (32 * 4 + 4) + (threadIdx.x % 32) * 4;

	//以对齐合并访存的模式从全局内存加载有限域乘法中间结果
	{
		uint8_t *read = dev_gfmult + dev_offset;
		uint8_t *write = smem + share_offset;

		#pragma unroll 4
		for (int i = 0; i < 4; i++)
		{
			*(uint32_t *)(write + i * ((BLOCK_SIZE / 32) * (32 * 4 + 4))) = (*(uint32_t *)(read + i * BLOCK_SIZE * 4));
		}
	}

	//确定线程束中，各个线程的共享内存读写地址
	uint8_t *smem_rw_pos = smem + (threadIdx.x / 32) * ((32 * 4 + 4) * 4) + dev_smem_offset[threadIdx.x % 32];
	
	__syncthreads();

	{
		uint8_t temp;
		uint8_t read[4];

		//暂存GF乘法中间结果
		uint8_t tid_gfmult[16];
		#pragma unroll 4
		for (int i = 0; i < 4; i++)
		{
			*(uint32_t *)(tid_gfmult + i * 4) = 0;
		}

		//查表法进行有限域乘法
		#pragma unroll 4
		for (int i = 0; i < 4; i++)
		{
			*(uint32_t *)read = *(uint32_t *)(smem_rw_pos + i * 4);

			#pragma unroll 4
			for (int j = 0; j < 4; j++)
			{
				temp = read[j] ^ constant_lenAC[i * 4 + j];

				#pragma unroll 16
				for (int k = 0; k < 16; k++)
				{
					tid_gfmult[k] ^= dev_gfmult_table[i * 4 + j][temp][k];
				}
			}
		}

		//每个线程与ency0异或生成最终的tag
		#pragma unroll 4
		for (int i = 0; i < 4; i++)
		{
			*(uint32_t *)(tid_gfmult + i * 4) ^= *(uint32_t *)(constant_ency0 + i * 4);
		}

		//将本数据块的有限域乘法的结果写回共享内存
		#pragma unroll 4
		for (int i = 0; i < 4; i++)
		{
			*(uint32_t *)(smem_rw_pos + i * 4) = *(uint32_t *)(tid_gfmult + i * 4);
		}
	}

	__syncthreads();

	//将共享内存中的乘法结果对齐合并访存的方式写回全局内存
	{
		uint8_t *write = dev_gfmult + dev_offset;
		uint8_t *read = smem + share_offset;
		#pragma unroll 4
		for (int i = 0; i < 4; i++)
		{
			*(uint32_t *)(write + i * BLOCK_SIZE * 4) = *(uint32_t *)(read + i * ((BLOCK_SIZE / 32) * (32 * 4 + 4)));
		}
	}
}


void Init_device_memory(device_memory *way, uint8_t add[16], uint8_t iv[12])
{
	//创建流
	for (int i = 0; i < STREAM_SIZE; i++)
	{
		cudaStreamCreate(&(way->stream[i]));
	}

	//初始化共享内存偏移量查找表内存空间
	cudaHostAlloc((void**)&(way->dev_smem_offset), 32 * sizeof(uint32_t), cudaHostAllocDefault);
	cudaMemcpy(way->dev_smem_offset, smem_offset, 32 * sizeof(uint32_t), cudaMemcpyHostToDevice);

	//将轮密钥拷贝到常量内存
	cudaMemcpyToSymbol(constant_sk, way->ctx.sk, 32 * sizeof(uint32_t));

	//初始化IV存储空间
	cudaHostAlloc((void**)&(way->dev_IV), PARTICLE_SIZE, cudaHostAllocDefault);

	//初始化每个线程的IV
	uint8_t *tempiv = (uint8_t *)malloc(PARTICLE_SIZE);
	
	for (int i = 0; i < PARTICLE_SIZE / 16; i++)
	{
		memcpy(tempiv + i * 16, iv, 12);
		//若需要对每个线程的IV不同则需要进行扩展
		//*(uint32_t *)(tempiv + i * 16 + 8) += i;
	}
	
	//将IV从主机内存拷贝到全局内存
	cudaMemcpy(way->dev_IV, tempiv, PARTICLE_SIZE, cudaMemcpyHostToDevice);
	free(tempiv);

	//初始化S盒内存空间
	cudaHostAlloc((void**)&(way->dev_SboxTable), 256, cudaHostAllocDefault);
	cudaMemcpy(way->dev_SboxTable, SboxTable, 256, cudaMemcpyHostToDevice);

	//分配数据输入空间
	cudaHostAlloc((void**)&(way->dev_input), PARTICLE_SIZE, cudaHostAllocDefault);

	//分配数据输出空间
	cudaHostAlloc((void**)&(way->dev_output), PARTICLE_SIZE, cudaHostAllocDefault);

	//加密全0密文块
	uint8_t y0[16];
	memset(y0, 0, 16);

	//将ency0拷贝到常量内存
	sm4_crypt_ecb(&way->ctx, 16, y0, ency0);
	cudaMemcpyToSymbol(constant_ency0, ency0, 16);

	//计算有限域乘法查找表
	computeTable(gfmult_table, ency0);

	//将有限域乘法表拷贝到全局内存
	cudaHostAlloc((void**)&(way->dev_gfmult_table), sizeof(gfmult_table), cudaHostAllocDefault);
	cudaMemcpy(way->dev_gfmult_table, gfmult_table, sizeof(gfmult_table), cudaMemcpyHostToDevice);

	//初始化有限域乘法表中间结果
	uint8_t temp[16];
	memset(temp, 0, 16);

	for (int i = 0; i < 16; i++)
	{
		temp[i] ^= add[i];
	}
	multi(gfmult_table, temp);

	uint8_t *gfmult_init = (uint8_t *)malloc(PARTICLE_SIZE);
	for (int i = 0; i < PARTICLE_SIZE / 16; i++)
	{
		memcpy(gfmult_init + i * 16, temp, 16);
	}

	//初始化有限域乘法输出空间
	cudaHostAlloc((void**)&(way->dev_gfmult), PARTICLE_SIZE, cudaHostAllocDefault);

	//将有限域乘法结果输出到全局内存
	cudaMemcpy(way->dev_gfmult, gfmult_init, PARTICLE_SIZE, cudaMemcpyHostToDevice);
	free(gfmult_init);
}

/*
**	主机接口函数:本函数完成设备内存的释放工作。
*/
void Free_device_memory(device_memory *way)
{
	for (int i = 0; i < STREAM_SIZE; i++)
	{
		//同步流
		cudaStreamSynchronize(way->stream[i]);
	}

	//释放全局内存
	cudaFreeHost(way->dev_gfmult_table);
	cudaFreeHost(way->dev_IV);
	cudaFreeHost(way->dev_SboxTable);

	cudaFreeHost(way->dev_smem_offset);

	cudaFreeHost(way->dev_input);
	cudaFreeHost(way->dev_output);
	cudaFreeHost(way->dev_gfmult);

	//释放流
	for (int i = 0; i < STREAM_SIZE; i++)
	{
		cudaStreamDestroy(way->stream[i]);
	}
}

/*
**	认证加密主机接口函数
**	counter:序号
**	input:明文输入
**	output:密文输出
*/
void sm4_gcm_enc(device_memory *way, uint32_t counter, uint8_t input[PARTICLE_SIZE], uint8_t output[PARTICLE_SIZE])
{
	dim3 grid(GRID_SIZE, 1, 1);
	dim3 block(BLOCK_SIZE, 1, 1);

	//将明文从主机内存拷贝到设备全局内存
	for (int i = 0; i < STREAM_SIZE; i++)
	{
		cudaMemcpyAsync(\
			way->dev_input + i * (PARTICLE_SIZE / STREAM_SIZE), \
			input + i * (PARTICLE_SIZE / STREAM_SIZE), \
			PARTICLE_SIZE / STREAM_SIZE, \
			cudaMemcpyHostToDevice, way->stream[i]);
	}

	//对明文数据块进行加密
	for (int i = 0; i < STREAM_SIZE; i++)
	{
		//对明文数据块进行加密
		kernal_enc <<< grid, block, 0, way->stream[i] >>> (way->dev_SboxTable, \
			way->dev_smem_offset, \
			counter, \
			way->dev_IV + i * (PARTICLE_SIZE / STREAM_SIZE), \
			way->dev_input + i * (PARTICLE_SIZE / STREAM_SIZE), \
			way->dev_output + i * (PARTICLE_SIZE / STREAM_SIZE));
	}

	//进行有限域乘法和加法运算
	for (int i = 0; i < STREAM_SIZE; i++)
	{
		kernal_gfmult <<< grid, block, 0, way->stream[i] >>> (way->dev_smem_offset, \
			(uint8_t(*)[256][16])(way->dev_gfmult_table), \
			way->dev_output + i * (PARTICLE_SIZE / STREAM_SIZE), \
			way->dev_gfmult + i * (PARTICLE_SIZE / STREAM_SIZE));
	}

	//将加密后的密文数据块从设备全局内存拷贝到主机内存
	for (int i = 0; i < STREAM_SIZE; i++)
	{
		cudaMemcpyAsync(output + i * (PARTICLE_SIZE / STREAM_SIZE), \
			way->dev_output + i * (PARTICLE_SIZE / STREAM_SIZE), \
			PARTICLE_SIZE / STREAM_SIZE, \
			cudaMemcpyDeviceToHost, way->stream[i]);
	}

	//流同步
	for (int i = 0; i < STREAM_SIZE; i++)
	{
		cudaStreamSynchronize(way->stream[i]);
	}
}

/*
**	认证解密主机接口函数
**	counter:加密序号
**	input:密文输入
**	output:明文输出
*/
void sm4_gcm_dec(device_memory *way, uint32_t counter, uint8_t input[PARTICLE_SIZE], uint8_t output[PARTICLE_SIZE])
{
	dim3 grid(GRID_SIZE, 1, 1);
	dim3 block(BLOCK_SIZE, 1, 1);

	//将密文从主机内存拷贝到设备全局内存
	for (int i = 0; i < STREAM_SIZE; i++)
	{
		cudaMemcpyAsync(\
			way->dev_input + i * (PARTICLE_SIZE / STREAM_SIZE), \
			input + i * (PARTICLE_SIZE / STREAM_SIZE), \
			PARTICLE_SIZE / STREAM_SIZE, \
			cudaMemcpyHostToDevice, way->stream[i]);
	}
	
	//进行有限域乘法和加法运算
	for (int i = 0; i < STREAM_SIZE; i++)
	{
		kernal_gfmult <<< grid, block, 0, way->stream[i] >>> (way->dev_smem_offset, \
			(uint8_t(*)[256][16])(way->dev_gfmult_table), \
			way->dev_input + i * (PARTICLE_SIZE / STREAM_SIZE), \
			way->dev_gfmult + i * (PARTICLE_SIZE / STREAM_SIZE));
	}
	
	//对密文数据块进行解密
	for (int i = 0; i < STREAM_SIZE; i++)
	{
		kernal_enc <<< grid, block, 0, way->stream[i] >>> (way->dev_SboxTable, \
			way->dev_smem_offset, \
			counter, \
			way->dev_IV + i * (PARTICLE_SIZE / STREAM_SIZE), \
			way->dev_input + i * (PARTICLE_SIZE / STREAM_SIZE), \
			way->dev_output + i * (PARTICLE_SIZE / STREAM_SIZE));
	}

	//将解密后的明文数据块从设备全局内存拷贝到主机内存
	for (int i = 0; i < STREAM_SIZE; i++)
	{
		//将解密后的明文数据块从设备全局内存拷贝到主机内存
		cudaMemcpyAsync(output + i * (PARTICLE_SIZE / STREAM_SIZE), \
			way->dev_output + i * (PARTICLE_SIZE / STREAM_SIZE), \
			PARTICLE_SIZE / STREAM_SIZE, \
			cudaMemcpyDeviceToHost, way->stream[i]);
	}

	//流同步
	for (int i = 0; i < STREAM_SIZE; i++)
	{
		cudaStreamSynchronize(way->stream[i]);
	}
}

/*
**	本主机接口函数生成最终的标签
**	length:密文数据块长度
**	tag:值结果参数，函数执行完毕将其填入tag的内存空间
*/
void sm4_gcm_final(device_memory *way, uint64_t length, uint8_t tag[PARTICLE_SIZE])
{
	uint8_t temp[16];
	/* eor (len(A)||len(C)) */
	uint64_t temp_len = (uint64_t)(16 * 8); // len(A) = (uint64_t)(add_len*8)
	for (int i = 1; i <= 16 / 2; i++)
	{
		temp[16 / 2 - i] = (uint8_t)temp_len;
		temp_len = temp_len >> 8;
	}
	length = length * 16;
	temp_len = (uint64_t)(length * 8); // len(C) = (uint64_t)(length*8)
	for (int i = 1; i <= 16 / 2; i++)
	{
		temp[16 - i] = (uint8_t)temp_len;
		temp_len = temp_len >> 8;
	}

	//初始化(len(A)||len(C))
	cudaMemcpyToSymbol(constant_lenAC, temp, 16);

	dim3 grid(GRID_SIZE, 1, 1);
	dim3 block(BLOCK_SIZE, 1, 1);

	//计算最终的GHASH结果
	kernal_final <<< grid, block, 0 >>> ((uint8_t(*)[256][16])(way->dev_gfmult_table), \
			way->dev_smem_offset, \
			way->dev_gfmult);
	
	//将每个线程的标签tag从全局内存拷贝回主机内存
	cudaMemcpyAsync(tag, way->dev_gfmult, PARTICLE_SIZE, cudaMemcpyDeviceToHost);
}

void printcuda(uint8_t *buffer, size_t len)
{
	printf("\n");
	for (int i = 0; i < len; i++)
	{
		if (i % (16) == 0)
		{
			printf("\n");
		}
		printf("0x%02X ", buffer[i]);
	}
	printf("\n");
}