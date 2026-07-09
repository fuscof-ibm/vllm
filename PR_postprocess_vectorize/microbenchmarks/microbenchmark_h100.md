python3 bench_copy_mamba_state_block.py 

MAIN

device: NVIDIA H100 80GB HBM3
model: Qwen/Qwen3.5-9B tp=1 num_spec=2
linear_layers=24 state_types=2 total_states=48
conv: dim=8192 state_len=5 dtype=torch.bfloat16 (=80.0 KiB/block)
temporal: inner=524288 dtype=torch.float32 (=2.00 MiB/block)
iters warmup=20 timed=200

reqs=1_conv=SD           grid=(1, 48)          copy/req= 49.88 MiB  avg=  963.25 us  BW~ 108.6 GB/s
reqs=4_conv=SD           grid=(4, 48)          copy/req= 49.88 MiB  avg= 1075.78 us  BW~ 388.9 GB/s
reqs=8_conv=SD           grid=(8, 48)          copy/req= 49.88 MiB  avg= 1174.75 us  BW~ 712.3 GB/s
reqs=16_conv=SD          grid=(16, 48)         copy/req= 49.88 MiB  avg= 1327.50 us  BW~1260.7 GB/s
reqs=32_conv=SD          grid=(32, 48)         copy/req= 49.88 MiB  avg= 1760.57 us  BW~1901.1 GB/s
reqs=64_conv=SD          grid=(64, 48)         copy/req= 49.88 MiB  avg= 3613.88 us  BW~1852.3 GB/s
reqs=128_conv=SD         grid=(128, 48)        copy/req= 49.88 MiB  avg= 6417.25 us  BW~2086.3 GB/s

total wall: 4.38s


PR (postprocess_vectorize)


device: NVIDIA H100 80GB HBM3
model: Qwen/Qwen3.5-9B tp=1 num_spec=2
linear_layers=24 state_types=2 total_states=48
conv: dim=8192 state_len=5 dtype=torch.bfloat16 (=80.0 KiB/block)
temporal: inner=524288 dtype=torch.float32 (=2.00 MiB/block)
iters warmup=20 timed=200

reqs=1_conv=SD           grid=(1, 48)          copy/req= 49.88 MiB  avg=  163.18 us  BW~ 641.0 GB/s
reqs=4_conv=SD           grid=(4, 48)          copy/req= 49.88 MiB  avg=  226.58 us  BW~1846.5 GB/s
reqs=8_conv=SD           grid=(8, 48)          copy/req= 49.88 MiB  avg=  338.41 us  BW~2472.6 GB/s
reqs=16_conv=SD          grid=(16, 48)         copy/req= 49.88 MiB  avg=  597.85 us  BW~2799.2 GB/s
reqs=32_conv=SD          grid=(32, 48)         copy/req= 49.88 MiB  avg= 1221.86 us  BW~2739.3 GB/s
reqs=64_conv=SD          grid=(64, 48)         copy/req= 49.88 MiB  avg= 2390.06 us  BW~2800.8 GB/s
reqs=128_conv=SD         grid=(128, 48)        copy/req= 49.88 MiB  avg= 4742.66 us  BW~2822.9 GB/s

total wall: 3.00s


