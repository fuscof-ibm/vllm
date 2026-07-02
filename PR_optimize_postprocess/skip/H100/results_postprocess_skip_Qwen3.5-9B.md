```
#!/bin/bash

export CUDA_VISIBLE_DEVICES="0"

MODEL_NAME="Qwen/Qwen3.5-9B"

VLLM_ENGINE_READY_TIMEOUT_S=1800 vllm serve $MODEL_NAME \
    --port 8007 \
    -tp 1 -pp 1 -dp 1 \
    --language-model-only \
    --reasoning-parser qwen3 \
    --enable-prefix-caching \
    --speculative-config '{"method":"qwen3_next_mtp","num_speculative_tokens":2}'
```


```
#!/usr/bin/env bash
# Decode-heavy random dataset: short prompt, long output.
# This is where the per-step CPU bubble matters most.
#
INPUT_LEN=256
OUTPUT_LEN=2048
NUM_PROMPTS=500

# Warmup (discarded) then sweep concurrency.
CONCURRENCIES=(1 4 8 16 32)

run_one() {
    local conc="$1"
    vllm bench serve \
	--port 8007 \
        --dataset-name random \
        --random-input-len "$INPUT_LEN" \
        --random-output-len "$OUTPUT_LEN" \
        --num-prompts "$NUM_PROMPTS" \
        --request-rate inf \
        --max-concurrency "$conc" \
        --ignore-eos \
        --percentile-metrics "ttft,tpot,itl,e2el" \
        --metric-percentiles "50,90,99"
}

# One warmup at low concurrency to populate caches / JIT Triton.

run_one 32
run_one 32
run_one 32

```



results for the last_precompiled   (4e5ca89cfe98121642d76b40e32a006f4d0fbf3b)

============ Serving Benchmark Result ============
Successful requests:                     500       
Failed requests:                         0         
Maximum request concurrency:             32        
Benchmark duration (s):                  251.64    
Total input tokens:                      128000    
Total generated tokens:                  1024000   
Request throughput (req/s):              1.99      
Output token throughput (tok/s):         4069.25   
Peak output token throughput (tok/s):    2357.00   
Peak concurrent requests:                39.00     
Total token throughput (tok/s):          4577.91   
---------------Time to First Token----------------
Mean TTFT (ms):                          126.35    
Median TTFT (ms):                        60.23     
P50 TTFT (ms):                           60.23     
P90 TTFT (ms):                           81.74     
P99 TTFT (ms):                           1073.46   
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          7.61      
Median TPOT (ms):                        7.28      
P50 TPOT (ms):                           7.28      
P90 TPOT (ms):                           9.50      
P99 TPOT (ms):                           13.35     
---------------Inter-token Latency----------------
Mean ITL (ms):                           14.02     
Median ITL (ms):                         13.81     
P50 ITL (ms):                            13.81     
P90 ITL (ms):                            15.48     
P99 ITL (ms):                            31.89     
----------------End-to-end Latency----------------
Mean E2EL (ms):                          15700.25  
Median E2EL (ms):                        15014.36  
P50 E2EL (ms):                           15014.36  
P90 E2EL (ms):                           19662.19  
P99 E2EL (ms):                           27377.52  
---------------Speculative Decoding---------------
Acceptance rate (%):                     42.17     
Acceptance length:                       1.84      
Drafts:                                  555464    
Draft tokens:                            1110928   
Accepted tokens:                         468425    
Per-position acceptance (%):
  Position 0:                            49.77     
  Position 1:                            34.56     
==================================================



============ Serving Benchmark Result ============
Successful requests:                     500       
Failed requests:                         0         
Maximum request concurrency:             32        
Benchmark duration (s):                  253.40    
Total input tokens:                      128000    
Total generated tokens:                  1024000   
Request throughput (req/s):              1.97      
Output token throughput (tok/s):         4040.98   
Peak output token throughput (tok/s):    2459.00   
Peak concurrent requests:                40.00     
Total token throughput (tok/s):          4546.10   
---------------Time to First Token----------------
Mean TTFT (ms):                          75.38     
Median TTFT (ms):                        61.15     
P50 TTFT (ms):                           61.15     
P90 TTFT (ms):                           81.03     
P99 TTFT (ms):                           298.26    
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          7.50      
Median TPOT (ms):                        7.23      
P50 TPOT (ms):                           7.23      
P90 TPOT (ms):                           9.41      
P99 TPOT (ms):                           13.03     
---------------Inter-token Latency----------------
Mean ITL (ms):                           13.94     
Median ITL (ms):                         13.79     
P50 ITL (ms):                            13.79     
P90 ITL (ms):                            15.47     
P99 ITL (ms):                            32.11     
----------------End-to-end Latency----------------
Mean E2EL (ms):                          15418.02  
Median E2EL (ms):                        14854.07  
P50 E2EL (ms):                           14854.07  
P90 E2EL (ms):                           19326.13  
P99 E2EL (ms):                           26902.44  
---------------Speculative Decoding---------------
Acceptance rate (%):                     43.03     
Acceptance length:                       1.86      
Drafts:                                  550281    
Draft tokens:                            1100562   
Accepted tokens:                         473624    
Per-position acceptance (%):
  Position 0:                            50.58     
  Position 1:                            35.49     
==================================================

============ Serving Benchmark Result ============
Successful requests:                     500       
Failed requests:                         0         
Maximum request concurrency:             32        
Benchmark duration (s):                  246.97    
Total input tokens:                      128000    
Total generated tokens:                  1024000   
Request throughput (req/s):              2.02      
Output token throughput (tok/s):         4146.21   
Peak output token throughput (tok/s):    2448.00   
Peak concurrent requests:                43.00     
Total token throughput (tok/s):          4664.49   
---------------Time to First Token----------------
Mean TTFT (ms):                          73.98     
Median TTFT (ms):                        60.84     
P50 TTFT (ms):                           60.84     
P90 TTFT (ms):                           79.93     
P99 TTFT (ms):                           298.12    
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          7.45      
Median TPOT (ms):                        7.17      
P50 TPOT (ms):                           7.17      
P90 TPOT (ms):                           9.13      
P99 TPOT (ms):                           12.69     
---------------Inter-token Latency----------------
Mean ITL (ms):                           13.96     
Median ITL (ms):                         13.79     
P50 ITL (ms):                            13.79     
P90 ITL (ms):                            15.30     
P99 ITL (ms):                            31.91     
----------------End-to-end Latency----------------
Mean E2EL (ms):                          15333.46  
Median E2EL (ms):                        14762.50  
P50 E2EL (ms):                           14762.50  
P90 E2EL (ms):                           18750.15  
P99 E2EL (ms):                           26274.42  
---------------Speculative Decoding---------------
Acceptance rate (%):                     43.70     
Acceptance length:                       1.87      
Drafts:                                  546368    
Draft tokens:                            1092736   
Accepted tokens:                         477547    
Per-position acceptance (%):
  Position 0:                            51.30     
  Position 1:                            36.11     
==================================================

results for the postprocess_skip (b199bb3b6c558166953619a340221b55d84805b7)

============ Serving Benchmark Result ============
Successful requests:                     500       
Failed requests:                         0         
Maximum request concurrency:             32        
Benchmark duration (s):                  253.50    
Total input tokens:                      128000    
Total generated tokens:                  1024000   
Request throughput (req/s):              1.97      
Output token throughput (tok/s):         4039.40   
Peak output token throughput (tok/s):    2417.00   
Peak concurrent requests:                40.00     
Total token throughput (tok/s):          4544.33   
---------------Time to First Token----------------
Mean TTFT (ms):                          94.66     
Median TTFT (ms):                        60.17     
P50 TTFT (ms):                           60.17     
P90 TTFT (ms):                           73.31     
P99 TTFT (ms):                           645.30    
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          7.60      
Median TPOT (ms):                        7.42      
P50 TPOT (ms):                           7.42      
P90 TPOT (ms):                           9.47      
P99 TPOT (ms):                           12.05     
---------------Inter-token Latency----------------
Mean ITL (ms):                           14.22     
Median ITL (ms):                         13.98     
P50 ITL (ms):                            13.98     
P90 ITL (ms):                            15.92     
P99 ITL (ms):                            33.88     
----------------End-to-end Latency----------------
Mean E2EL (ms):                          15658.89  
Median E2EL (ms):                        15256.29  
P50 E2EL (ms):                           15256.29  
P90 E2EL (ms):                           19556.87  
P99 E2EL (ms):                           24730.44  
---------------Speculative Decoding---------------
Acceptance rate (%):                     43.54     
Acceptance length:                       1.87      
Drafts:                                  547308    
Draft tokens:                            1094616   
Accepted tokens:                         476574    
Per-position acceptance (%):
  Position 0:                            51.15     
  Position 1:                            35.92     
==================================================

============ Serving Benchmark Result ============
Successful requests:                     500       
Failed requests:                         0         
Maximum request concurrency:             32        
Benchmark duration (s):                  260.60    
Total input tokens:                      128000    
Total generated tokens:                  1024000   
Request throughput (req/s):              1.92      
Output token throughput (tok/s):         3929.40   
Peak output token throughput (tok/s):    2400.00   
Peak concurrent requests:                39.00     
Total token throughput (tok/s):          4420.57   
---------------Time to First Token----------------
Mean TTFT (ms):                          74.81     
Median TTFT (ms):                        60.91     
P50 TTFT (ms):                           60.91     
P90 TTFT (ms):                           80.50     
P99 TTFT (ms):                           297.34    
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          7.74      
Median TPOT (ms):                        7.42      
P50 TPOT (ms):                           7.42      
P90 TPOT (ms):                           9.78      
P99 TPOT (ms):                           13.01     
---------------Inter-token Latency----------------
Mean ITL (ms):                           14.22     
Median ITL (ms):                         13.98     
P50 ITL (ms):                            13.98     
P90 ITL (ms):                            15.95     
P99 ITL (ms):                            32.74     
----------------End-to-end Latency----------------
Mean E2EL (ms):                          15925.75  
Median E2EL (ms):                        15269.95  
P50 E2EL (ms):                           15269.95  
P90 E2EL (ms):                           20083.63  
P99 E2EL (ms):                           26687.49  
---------------Speculative Decoding---------------
Acceptance rate (%):                     41.88     
Acceptance length:                       1.84      
Drafts:                                  557177    
Draft tokens:                            1114354   
Accepted tokens:                         466714    
Per-position acceptance (%):
  Position 0:                            49.41     
  Position 1:                            34.36     
==================================================

============ Serving Benchmark Result ============
Successful requests:                     500       
Failed requests:                         0         
Maximum request concurrency:             32        
Benchmark duration (s):                  256.96    
Total input tokens:                      128000    
Total generated tokens:                  1024000   
Request throughput (req/s):              1.95      
Output token throughput (tok/s):         3985.12   
Peak output token throughput (tok/s):    2400.00   
Peak concurrent requests:                39.00     
Total token throughput (tok/s):          4483.26   
---------------Time to First Token----------------
Mean TTFT (ms):                          74.08     
Median TTFT (ms):                        61.24     
P50 TTFT (ms):                           61.24     
P90 TTFT (ms):                           81.72     
P99 TTFT (ms):                           299.89    
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          7.68      
Median TPOT (ms):                        7.38      
P50 TPOT (ms):                           7.38      
P90 TPOT (ms):                           9.56      
P99 TPOT (ms):                           12.49     
---------------Inter-token Latency----------------
Mean ITL (ms):                           14.22     
Median ITL (ms):                         13.96     
P50 ITL (ms):                            13.96     
P90 ITL (ms):                            15.94     
P99 ITL (ms):                            32.94     
----------------End-to-end Latency----------------
Mean E2EL (ms):                          15786.65  
Median E2EL (ms):                        15178.06  
P50 E2EL (ms):                           15178.06  
P90 E2EL (ms):                           19704.78  
P99 E2EL (ms):                           25618.73  
---------------Speculative Decoding---------------
Acceptance rate (%):                     42.66     
Acceptance length:                       1.85      
Drafts:                                  552479    
Draft tokens:                            1104958   
Accepted tokens:                         471425    
Per-position acceptance (%):
  Position 0:                            50.24     
  Position 1:                            35.09     
==================================================

results for the postprocess_skip (eb67c9e80ff212603372c1b273e7c63494a9d6c9)


============ Serving Benchmark Result ============
Successful requests:                     500       
Failed requests:                         0         
Maximum request concurrency:             32        
Benchmark duration (s):                  256.47    
Total input tokens:                      128000    
Total generated tokens:                  1024000   
Request throughput (req/s):              1.95      
Output token throughput (tok/s):         3992.72   
Peak output token throughput (tok/s):    2400.00   
Peak concurrent requests:                39.00     
Total token throughput (tok/s):          4491.81   
---------------Time to First Token----------------
Mean TTFT (ms):                          92.60     
Median TTFT (ms):                        59.57     
P50 TTFT (ms):                           59.57     
P90 TTFT (ms):                           72.52     
P99 TTFT (ms):                           634.80    
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          7.58      
Median TPOT (ms):                        7.36      
P50 TPOT (ms):                           7.36      
P90 TPOT (ms):                           9.30      
P99 TPOT (ms):                           12.83     
---------------Inter-token Latency----------------
Mean ITL (ms):                           14.19     
Median ITL (ms):                         13.96     
P50 ITL (ms):                            13.96     
P90 ITL (ms):                            15.89     
P99 ITL (ms):                            33.51     
----------------End-to-end Latency----------------
Mean E2EL (ms):                          15603.77  
Median E2EL (ms):                        15144.76  
P50 E2EL (ms):                           15144.76  
P90 E2EL (ms):                           19222.16  
P99 E2EL (ms):                           26537.63  
---------------Speculative Decoding---------------
Acceptance rate (%):                     43.70     
Acceptance length:                       1.87      
Drafts:                                  546364    
Draft tokens:                            1092728   
Accepted tokens:                         477551    
Per-position acceptance (%):
  Position 0:                            51.30     
  Position 1:                            36.11     
==================================================

============ Serving Benchmark Result ============
Successful requests:                     500       
Failed requests:                         0         
Maximum request concurrency:             32        
Benchmark duration (s):                  253.98    
Total input tokens:                      128000    
Total generated tokens:                  1024000   
Request throughput (req/s):              1.97      
Output token throughput (tok/s):         4031.75   
Peak output token throughput (tok/s):    2400.00   
Peak concurrent requests:                40.00     
Total token throughput (tok/s):          4535.72   
---------------Time to First Token----------------
Mean TTFT (ms):                          74.36     
Median TTFT (ms):                        60.92     
P50 TTFT (ms):                           60.92     
P90 TTFT (ms):                           72.83     
P99 TTFT (ms):                           311.05    
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          7.69      
Median TPOT (ms):                        7.45      
P50 TPOT (ms):                           7.45      
P90 TPOT (ms):                           9.69      
P99 TPOT (ms):                           12.58     
---------------Inter-token Latency----------------
Mean ITL (ms):                           14.18     
Median ITL (ms):                         13.96     
P50 ITL (ms):                            13.96     
P90 ITL (ms):                            15.93     
P99 ITL (ms):                            32.95     
----------------End-to-end Latency----------------
Mean E2EL (ms):                          15810.51  
Median E2EL (ms):                        15307.52  
P50 E2EL (ms):                           15307.52  
P90 E2EL (ms):                           19877.89  
P99 E2EL (ms):                           25805.29  
---------------Speculative Decoding---------------
Acceptance rate (%):                     42.25     
Acceptance length:                       1.84      
Drafts:                                  554974    
Draft tokens:                            1109948   
Accepted tokens:                         468940    
Per-position acceptance (%):
  Position 0:                            49.76     
  Position 1:                            34.73     
==================================================


============ Serving Benchmark Result ============
Successful requests:                     500       
Failed requests:                         0         
Maximum request concurrency:             32        
Benchmark duration (s):                  258.09    
Total input tokens:                      128000    
Total generated tokens:                  1024000   
Request throughput (req/s):              1.94      
Output token throughput (tok/s):         3967.68   
Peak output token throughput (tok/s):    2400.00   
Peak concurrent requests:                39.00     
Total token throughput (tok/s):          4463.64   
---------------Time to First Token----------------
Mean TTFT (ms):                          67.53     
Median TTFT (ms):                        50.01     
P50 TTFT (ms):                           50.01     
P90 TTFT (ms):                           70.90     
P99 TTFT (ms):                           301.27    
-----Time per Output Token (excl. 1st token)------
Mean TPOT (ms):                          7.76      
Median TPOT (ms):                        7.47      
P50 TPOT (ms):                           7.47      
P90 TPOT (ms):                           9.76      
P99 TPOT (ms):                           13.54     
---------------Inter-token Latency----------------
Mean ITL (ms):                           14.18     
Median ITL (ms):                         13.95     
P50 ITL (ms):                            13.95     
P90 ITL (ms):                            15.83     
P99 ITL (ms):                            33.65     
----------------End-to-end Latency----------------
Mean E2EL (ms):                          15947.36  
Median E2EL (ms):                        15356.27  
P50 E2EL (ms):                           15356.27  
P90 E2EL (ms):                           20138.48  
P99 E2EL (ms):                           27759.37  
---------------Speculative Decoding---------------
Acceptance rate (%):                     41.42     
Acceptance length:                       1.83      
Drafts:                                  560032    
Draft tokens:                            1120064   
Accepted tokens:                         463880    
Per-position acceptance (%):
  Position 0:                            48.96     
  Position 1:                            33.88     
==================================================

