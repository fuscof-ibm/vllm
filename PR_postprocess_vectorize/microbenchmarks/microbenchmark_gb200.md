python3 bench_copy_mamba_state_block.py


MAIN


device: NVIDIA GB200
model: Qwen/Qwen3.5-9B tp=1 num_spec=2
linear_layers=24 state_types=2 total_states=48
conv: dim=8192 state_len=5 dtype=torch.bfloat16 (=80.0 KiB/block)
temporal: inner=524288 dtype=torch.float32 (=2.00 MiB/block)
iters warmup=20 timed=200

reqs=1_conv=SD           grid=(1, 48)          copy/req= 49.88 MiB  avg= 1417.62 us  BW~  73.8 GB/s
reqs=4_conv=SD           grid=(4, 48)          copy/req= 49.88 MiB  avg= 1493.72 us  BW~ 280.1 GB/s
reqs=8_conv=SD           grid=(8, 48)          copy/req= 49.88 MiB  avg= 1527.27 us  BW~ 547.9 GB/s
reqs=16_conv=SD          grid=(16, 48)         copy/req= 49.88 MiB  avg= 1642.00 us  BW~1019.2 GB/s
reqs=32_conv=SD          grid=(32, 48)         copy/req= 49.88 MiB  avg= 1934.67 us  BW~1730.0 GB/s
reqs=64_conv=SD          grid=(64, 48)         copy/req= 49.88 MiB  avg= 3240.21 us  BW~2065.9 GB/s
reqs=128_conv=SD         grid=(128, 48)        copy/req= 49.88 MiB  avg= 5136.81 us  BW~2606.3 GB/s

total wall: 4.50s


PR (postprocess_vectorize)


device: NVIDIA GB200
model: Qwen/Qwen3.5-9B tp=1 num_spec=2
linear_layers=24 state_types=2 total_states=48
conv: dim=8192 state_len=5 dtype=torch.bfloat16 (=80.0 KiB/block)
temporal: inner=524288 dtype=torch.float32 (=2.00 MiB/block)
iters warmup=20 timed=200

reqs=1_conv=SD           grid=(1, 48)          copy/req= 49.88 MiB  avg=  218.20 us  BW~ 479.4 GB/s
reqs=4_conv=SD           grid=(4, 48)          copy/req= 49.88 MiB  avg=  255.02 us  BW~1640.6 GB/s
reqs=8_conv=SD           grid=(8, 48)          copy/req= 49.88 MiB  avg=  274.35 us  BW~3050.0 GB/s
reqs=16_conv=SD          grid=(16, 48)         copy/req= 49.88 MiB  avg=  353.22 us  BW~4737.9 GB/s
reqs=32_conv=SD          grid=(32, 48)         copy/req= 49.88 MiB  avg=  567.24 us  BW~5900.6 GB/s
reqs=64_conv=SD          grid=(64, 48)         copy/req= 49.88 MiB  avg= 1295.81 us  BW~5165.9 GB/s
reqs=128_conv=SD         grid=(128, 48)        copy/req= 49.88 MiB  avg= 2313.11 us  BW~5788.0 GB/s

total wall: 1.64s

