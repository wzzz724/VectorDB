#include <stdint.h>
#include <iostream>
#include <hls_stream.h>

typedef unsigned char int_8;

struct Embedding {
    int_8 item[128 + 1]; // 最后一位是tag
};

static int inner_product(Embedding A, Embedding B) {
    int res = 0;
    for (int i = 0; i < 128; i++) {
#pragma HLS UNROLL factor=128
        res += A.item[i] * B.item[i];
    }
    return res;
}

void compute(Embedding *in, Embedding *out, Embedding *k_centers, int K, int size, int* counter) {
	for (int i = 0; i < size; i++) {
		if (i < K) {
			in[i].item[128] = i;
			counter[i]++;
			k_centers[i] = in[i];
		} else {
			int max_j = 0, max = -1;
			for (int j = 0; j < K; j++) {
				int ip = inner_product(in[i], k_centers[j]);
				if (ip > max) {
					max_j = j;
					max = ip;
				}
			}
			in[i][128] = max_j;
			for (int _ = 0; _ < 128; _++) {
				k_centers[max_j].item[_] = (k_centers[max_j].item[_] * counter[max_j] + in[i].item[_]) / (counter[max_j] + 1);
			}
			counter[max_j]++;
		}
	}
}

void index(Embedding *in, Embedding *out, Embedding *k_centers, int K, int size){

#pragma HLS INTERFACE m_axi port = in bundle = gmem0
#pragma HLS INTERFACE m_axi port = out bundle = gmem1
#pragma HLS INTERFACE m_axi port = centers bundle = gmem2

	Embedding centers[K];
	int counter[K];

	compute(in, out, k_centers, K, size, counter);
}
