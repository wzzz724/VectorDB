#include <stdint.h>
#include <iostream>
#include <hls_stream.h>

typedef unsigned char int_8;

struct Embedding {
    int_8 item[32];
};

// TRIPCOUNT identifier
// const int c_size = 1024;
// TopK identifier
const int M = 10;
const int K = (1 << M);
// merger series Latency: 2^M - 1 + M Cycles
// const int max_sorter_size = c_size + K - 1 + M;
// // top-K merger Latency: 2^M + M Cycles
// const int max_merger_size = max_sorter_size + 1;

struct Deque {
    int data[1024 + 1];
    int head = 0, tail = 0, size = 0;

    bool empty() { return size == 0; }
    int front() { return data[head]; } 
    void push(int x) {
        data[tail] = x;
        tail = tail == 1024 ? 1 : tail + 1;
        size ++;
    }
    void pop_front() {
        head = head == 1024 ? 1 : head + 1;
        size --;
    }
    void pop_back() {
        tail = tail == 0 ? 1024 : tail - 1;
        size --;
    }
    void clear() {
        head = 0;
        tail = 0;
        size = 0;
    }
};
// top-K selection
static int r[M] = {0};
static Deque A0, A1, A2, A3, A4, A5, A6, A7, A8, A9;
static Deque B0, B1, B2, B3, B4, B5, B6, B7, B8, B9;
static int A_pop_count[M] = {0};
static int sort_r = 0;
static Deque sort_A, sort_B, sort_C;
// Filter
static int min_topK = -1;
// static Deque filter;

static int inner_product(
    Embedding A,
    Embedding B
) {
in_p:
    int res = 0;
    for (int i = 0; i < 32; i++) {
#pragma HLS UNROLL factor=32
        res += A.item[i] * B.item[i];
    }
    return res;
}

// 每个向量都是32*4=128个int8，分割为4部分即in1~in4。size是向量的总个数
static void similarity_compute_directly_read_from_hbm(
    Embedding* in1, Embedding* in2, Embedding* in3, Embedding* in4,
    hls::stream<int>& calcStream,
    int size,
    Embedding* query
) {
similarity_calc_directly_from_hbm:
    for (int i = 0; i < size; i++) {
// #pragma HLS LOOP_TRIPCOUNT min = c_size max = c_size
        int res = 0;
        res += inner_product(in1[i], query[0]);
        res += inner_product(in2[i], query[1]);
        res += inner_product(in3[i], query[2]);
        res += inner_product(in4[i], query[3]);
        calcStream << res;
    }
}

static void similarity_compute(
    hls::stream<Embedding>& inStream1, 
    hls::stream<Embedding>& inStream2, 
    hls::stream<Embedding>& inStream3, 
    hls::stream<Embedding>& inStream4, 
    hls::stream<int>& calcStream,
    int size,
    Embedding* query
) {
similarity_calc:
    for (int i = 0; i < size; i++) {
// #pragma HLS LOOP_TRIPCOUNT min = c_size max = c_size
        int res = 0;
        res += inner_product(inStream1.read(), query[0]);
        res += inner_product(inStream2.read(), query[1]);
        res += inner_product(inStream3.read(), query[2]);
        res += inner_product(inStream4.read(), query[3]);
        calcStream << res;
    }
}

static void data_filter(
    hls::stream<int>& calcStream1, hls::stream<int>& calcStream2,
    hls::stream<int>& calcStream3, hls::stream<int>& calcStream4,
    hls::stream<int>& calcStream5, hls::stream<int>& calcStream6,
    hls::stream<int>& calcStream7, hls::stream<int>& calcStream8,
    hls::stream<int, 2 * K>* dt,
    hls::stream<int>& outStream,
    hls::stream<int>& feedbackStream,
    int size
) {
data_filt:
    for (int i = 0; i < size; i++) {
// #pragma HLS LOOP_TRIPCOUNT min = c_size max = c_size
#pragma HLS PIPELINE II=1
        // Each Round produce 1 score
        int score_1 = calcStream1.read();
        int score_2 = calcStream2.read();
        int score_3 = calcStream3.read();
        int score_4 = calcStream4.read();
        int score_5 = calcStream5.read();
        int score_6 = calcStream6.read();
        int score_7 = calcStream7.read();
        int score_8 = calcStream8.read();
       // outStream << score_1;
        int temp_min = -1;
        if (i >= K + 2 * M + 2) {
            temp_min = feedbackStream.read();
        }

        if (score_1 > temp_min) dt[0] << score_1;
        if (score_2 > temp_min) dt[1] << score_2;
        if (score_3 > temp_min) dt[2] << score_3;
        if (score_4 > temp_min) dt[3] << score_4;
        if (score_5 > temp_min) dt[4] << score_5;
        if (score_6 > temp_min) dt[5] << score_6;
        if (score_7 > temp_min) dt[6] << score_7;
        if (score_8 > temp_min) dt[7] << score_8;

        int mask = (dt[7].empty() << 0) +  (dt[6].empty() << 1) +  (dt[5].empty() << 2) +  (dt[4].empty() << 3) + 
         (dt[3].empty() << 4) +  (dt[2].empty() << 5) +  (dt[1].empty() << 6) + (dt[0].empty() << 7);

        if (mask == 0) outStream << score_1;
        else {
            int highestBitPosition = -1; // initialize the highest bit position
            while (mask != 0) {
                mask >>= 1; // shift right
                highestBitPosition++; // increase the highest bit position
            }
            outStream << dt[highestBitPosition].read();
        }
    }
}

static void k_sorter(
    int m,
    hls::stream<int>& inStream,
    hls::stream<int>& outStream,
    Deque& A,
    Deque& B,
    int size
) {
K_sorter_2:
    for (int i = 0; i < size + (1 << (m + 1)) + m; i ++) {
// #pragma HLS LOOP_TRIPCOUNT min = c_size max = max_sorter_size
        if (i >= (1 << m) + m - 1) { // The first input is received
            int output = -1;
            int k = (1 << m);
            int rm = r[m], af = A.front(), bf = B.front();
            bool ae = A.empty(), be = B.empty();
            int apc = A_pop_count[m];
            if (rm > k) { // The first output is produced
                if (ae) {
                    output = bf;
                    B.pop_front();
                } else if (be) {
                    output = af;
                    A.pop_front();
                    A_pop_count[m] = apc + 1;
                } else if (af > bf && apc < k) {
                    output = af;
                    A.pop_front();
                    A_pop_count[m] = apc + 1;
                } else {
                    output = bf;
                    B.pop_front();
                }
                outStream << output;
            }                       
            if (i < size + (1 << m) + m - 1) { // still have input
                int input = inStream.read();
                if ((rm & (2 * k - 1)) < k) { // -> before: r[m] % (2 * k) < k
                    A.push(input);
                } else {
                    B.push(input);
                }
            }       
            if (rm - k > 0 && ((rm - k) & (2 * k - 1) == 0)) { // before : ((r[m] - k) % (2 * k) == 0)
                A_pop_count[m] = 0;
            }
            r[m] += 1;
        }
    }
}

static void topk_merger(
    hls::stream<int>& inStream,
    hls::stream<int>& outStream,
    hls::stream<int>& feedbackStream,
    int size
) {
topk_merge:
    for (int i = 0; i < size + K + M; i ++) {
// #pragma HLS LOOP_TRIPCOUNT min = c_size max = max_merger_size
        if (i >= K - 1 + M) { // The first input is received
            int res = -1;
            int k = K;
            int str = sort_r;
            int af = sort_A.front(), bf = sort_B.front(), cf = sort_C.front();
            if (str > 0) { // output
                if (((str - 1) & (2 * k - 1)) < k) { // before (str - 1) % (2 * k) < k
                    if (str - 1 < k) {
                        res = cf;
                        sort_A.push(res);
                        sort_C.pop_front();
                    } else {
                        if (bf > cf) {
                            res = bf;
                            sort_A.push(res);
                            sort_B.pop_front();
                        } else {
                            res = cf;
                            sort_A.push(res);
                            sort_C.pop_front();
                            sort_B.pop_back();
                        }
                        if ((str & (k - 1)) == 0) { // before str % k == 0
                            sort_C.clear();
                        }
                    }
                } else {
                    if (af > cf) {
                        res = af;
                        sort_B.push(res);
                        sort_A.pop_front();
                    } else {
                        res = cf;
                        sort_B.push(res);
                        sort_C.pop_front();
                        sort_A.pop_back();
                    }
                    if ((str & (k - 1)) == 0) {
                        sort_C.clear();
                    } 
                }
                if (res > min_topK) min_topK = res;
                if (i >= K + M && i <= size - M - 3) feedbackStream << min_topK;
                outStream << res;
            }
            if (i < size + K + M - 1) { // still have input
                sort_C.push(inStream.read());
                sort_r += 1;
            }
        }
    }
}

static void write_k_merger_result(
    // int* out, 
    hls::stream<int>& mergerStream,
    int size
) {
k_merger_mem_wr:
    for (int i = 0; i < size + K + M; i++) {
// #pragma HLS LOOP_TRIPCOUNT min = max_merger_size max = max_merger_size
        if (i >= K + M) {
            int a = mergerStream.read();
            // if (i >= size + K + M - 1024)
            //     out[i - (size + K + M - 1024)] = a;
        }
    }
}

extern "C" {

void vadd(
    Embedding* in1, Embedding* in2, Embedding* in3, Embedding* in4,
    Embedding* in5, Embedding* in6, Embedding* in7, Embedding* in8,
    Embedding* in9, Embedding* in10, Embedding* in11, Embedding* in12,
    Embedding* in13, Embedding* in14, Embedding* in15, Embedding* in16,
    Embedding* in17, Embedding* in18, Embedding* in19, Embedding* in20,
    Embedding* in21, Embedding* in22, Embedding* in23, Embedding* in24,
    Embedding* in25, Embedding* in26, Embedding* in27, Embedding* in28,
    Embedding* in29, Embedding* in30, Embedding* in31, Embedding* in32,
    // Embedding* out1, Embedding* out2, Embedding* out3, Embedding* out4, 
    // int* out1,
    int size
    // int_8 *fpga_query
) {

    static hls::stream<Embedding> inStream[32];
    static hls::stream<int> calcStream[8];
    static hls::stream<int, 2 * K> dt[8];
    static hls::stream<int> filterStream;
    static hls::stream<int> sorterStream[M];
    static hls::stream<int> mergerStream;
    static hls::stream<int> feedbackStream;

    // A Query from Userspace (DMA in FPGA)
    Embedding query[4];
    for (size_t i = 0; i < 4; i++) {
        for (size_t j = 0; j < 32; j++) {
            query[i].item[j] = 1;
        }
    }


#pragma HLS INTERFACE m_axi port = in1 bundle = gmem0
#pragma HLS INTERFACE m_axi port = in2 bundle = gmem1
#pragma HLS INTERFACE m_axi port = in3 bundle = gmem2
#pragma HLS INTERFACE m_axi port = in4 bundle = gmem3
#pragma HLS INTERFACE m_axi port = in5 bundle = gmem4
#pragma HLS INTERFACE m_axi port = in6 bundle = gmem5
#pragma HLS INTERFACE m_axi port = in7 bundle = gmem6
#pragma HLS INTERFACE m_axi port = in8 bundle = gmem7
#pragma HLS INTERFACE m_axi port = in9 bundle = gmem8
#pragma HLS INTERFACE m_axi port = in10 bundle = gmem9
#pragma HLS INTERFACE m_axi port = in11 bundle = gmem10
#pragma HLS INTERFACE m_axi port = in12 bundle = gmem11
#pragma HLS INTERFACE m_axi port = in13 bundle = gmem12
#pragma HLS INTERFACE m_axi port = in14 bundle = gmem13
#pragma HLS INTERFACE m_axi port = in15 bundle = gmem14
#pragma HLS INTERFACE m_axi port = in16 bundle = gmem15
#pragma HLS INTERFACE m_axi port = in17 bundle = gmem16
#pragma HLS INTERFACE m_axi port = in18 bundle = gmem17
#pragma HLS INTERFACE m_axi port = in19 bundle = gmem18
#pragma HLS INTERFACE m_axi port = in20 bundle = gmem19
#pragma HLS INTERFACE m_axi port = in21 bundle = gmem20
#pragma HLS INTERFACE m_axi port = in22 bundle = gmem21
#pragma HLS INTERFACE m_axi port = in23 bundle = gmem22
#pragma HLS INTERFACE m_axi port = in24 bundle = gmem23
#pragma HLS INTERFACE m_axi port = in25 bundle = gmem24
#pragma HLS INTERFACE m_axi port = in26 bundle = gmem25
#pragma HLS INTERFACE m_axi port = in27 bundle = gmem26
#pragma HLS INTERFACE m_axi port = in28 bundle = gmem27
#pragma HLS INTERFACE m_axi port = in29 bundle = gmem28
#pragma HLS INTERFACE m_axi port = in30 bundle = gmem29
#pragma HLS INTERFACE m_axi port = in31 bundle = gmem30
#pragma HLS INTERFACE m_axi port = in32 bundle = gmem31
// #pragma HLS INTERFACE m_axi port = out1 bundle = gmem16
// #pragma HLS array_partition variable=A0 complete

// #pragma HLS bind_storage variable=min_topK type=ram_t2p
// #pragma HLS array_partition variable=A_pop_count type=complete

#pragma HLS dataflow
    // dataflow pragma instruct compiler to run following three APIs in parallel
    // 8路并行计算相似度
    similarity_compute_directly_read_from_hbm(
        in1, in2, in3, in4,
        calcStream[0], size, query
    );
    similarity_compute_directly_read_from_hbm(
        in5, in6, in7, in8,
        calcStream[1], size, query
    );
    similarity_compute_directly_read_from_hbm(
        in9, in10, in11, in12,
        calcStream[2], size, query
    );
    similarity_compute_directly_read_from_hbm(
        in13, in14, in15, in16,
        calcStream[3], size, query
    );
    similarity_compute_directly_read_from_hbm(
        in17, in18, in19, in20,
        calcStream[4], size, query
    );
    similarity_compute_directly_read_from_hbm(
        in21, in22, in23, in24,
        calcStream[5], size, query
    );
    similarity_compute_directly_read_from_hbm(
        in25, in26, in27, in28,
        calcStream[6], size, query
    );
    similarity_compute_directly_read_from_hbm(
        in29, in30, in31, in32,
        calcStream[7], size, query
    );

    // write_result(out1, calcStream[0], size);
    data_filter(
        calcStream[0], calcStream[1], calcStream[2], calcStream[3],
        calcStream[4], calcStream[5], calcStream[6], calcStream[7],
        // dt1, dt2, dt3, dt4,
        // dt5, dt6, dt7, dt8,
        dt,
        filterStream, feedbackStream, size
    );

    k_sorter(0, filterStream, sorterStream[0], A0, B0, size);

    k_sorter(1, sorterStream[0], sorterStream[1], A1, B1, size);
    k_sorter(2, sorterStream[1], sorterStream[2], A2, B2, size);
    k_sorter(3, sorterStream[2], sorterStream[3], A3, B3, size);
    k_sorter(4, sorterStream[3], sorterStream[4], A4, B4, size);
    k_sorter(5, sorterStream[4], sorterStream[5], A5, B5, size);
    k_sorter(6, sorterStream[5], sorterStream[6], A6, B6, size);
    k_sorter(7, sorterStream[6], sorterStream[7], A7, B7, size);
    k_sorter(8, sorterStream[7], sorterStream[8], A8, B8, size);
    k_sorter(9, sorterStream[8], sorterStream[9], A9, B9, size);
    topk_merger(sorterStream[9], mergerStream, feedbackStream, size);
    write_k_merger_result(mergerStream, size);
}

}
