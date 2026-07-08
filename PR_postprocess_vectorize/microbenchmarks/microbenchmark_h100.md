python3 benchmark/bench_copy_mamba_state_block.py 

MAIN

device: NVIDIA H100 80GB HBM3
model: Qwen/Qwen3.5-9B tp=1 num_spec=2
linear_layers=24 state_types=2 total_states=48
conv: dim=8192 state_len=5 (=80.0 KiB/block)
temporal: inner=524288 (=1.00 MiB/block)
iters warmup=20 timed=200

reqs=1_conv=SD           grid=(1, 48)          copy/req= 25.88 MiB  avg=  491.97 us  BW~ 110.3 GB/s
reqs=4_conv=SD           grid=(4, 48)          copy/req= 25.88 MiB  avg=  541.85 us  BW~ 400.6 GB/s
reqs=8_conv=SD           grid=(8, 48)          copy/req= 25.88 MiB  avg=  588.92 us  BW~ 737.1 GB/s
reqs=16_conv=SD          grid=(16, 48)         copy/req= 25.88 MiB  avg=  669.82 us  BW~1296.2 GB/s
reqs=32_conv=SD          grid=(32, 48)         copy/req= 25.88 MiB  avg=  901.88 us  BW~1925.4 GB/s
reqs=64_conv=SD          grid=(64, 48)         copy/req= 25.88 MiB  avg= 1829.67 us  BW~1898.1 GB/s
reqs=128_conv=SD         grid=(128, 48)        copy/req= 25.88 MiB  avg= 3288.01 us  BW~2112.5 GB/s

total wall: 2.24s

PR (postprocess_vectorize)

device: NVIDIA H100 80GB HBM3
model: Qwen/Qwen3.5-9B tp=1 num_spec=2
linear_layers=24 state_types=2 total_states=48
conv: dim=8192 state_len=5 (=80.0 KiB/block)
temporal: inner=524288 (=1.00 MiB/block)
iters warmup=20 timed=200

reqs=1_conv=SD           grid=(1, 48)          copy/req= 25.88 MiB  avg=   84.50 us  BW~ 642.2 GB/s
reqs=4_conv=SD           grid=(4, 48)          copy/req= 25.88 MiB  avg=  117.49 us  BW~1847.4 GB/s
reqs=8_conv=SD           grid=(8, 48)          copy/req= 25.88 MiB  avg=  178.58 us  BW~2430.9 GB/s
reqs=16_conv=SD          grid=(16, 48)         copy/req= 25.88 MiB  avg=  316.76 us  BW~2741.0 GB/s
reqs=32_conv=SD          grid=(32, 48)         copy/req= 25.88 MiB  avg=  638.97 us  BW~2717.6 GB/s
reqs=64_conv=SD          grid=(64, 48)         copy/req= 25.88 MiB  avg= 1247.38 us  BW~2784.1 GB/s
reqs=128_conv=SD         grid=(128, 48)        copy/req= 25.88 MiB  avg= 2459.43 us  BW~2824.1 GB/s

total wall: 1.52s
