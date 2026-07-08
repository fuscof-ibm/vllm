
python benchmark/bench_copy_mamba_state_block.py 

MAIN:

device: NVIDIA GB200
model: Qwen/Qwen3.5-9B tp=1 num_spec=2
linear_layers=24 state_types=2 total_states=48
conv: dim=8192 state_len=5 (=80.0 KiB/block)
temporal: inner=524288 (=1.00 MiB/block)
iters warmup=20 timed=200

reqs=1_conv=SD           grid=(1, 48)          copy/req= 25.88 MiB  avg=  365.31 us  BW~ 148.5 GB/s
reqs=4_conv=SD           grid=(4, 48)          copy/req= 25.88 MiB  avg=  743.50 us  BW~ 291.9 GB/s
reqs=8_conv=SD           grid=(8, 48)          copy/req= 25.88 MiB  avg=  761.45 us  BW~ 570.1 GB/s
reqs=16_conv=SD          grid=(16, 48)         copy/req= 25.88 MiB  avg=  822.26 us  BW~1055.9 GB/s
reqs=32_conv=SD          grid=(32, 48)         copy/req= 25.88 MiB  avg=  986.63 us  BW~1760.0 GB/s
reqs=64_conv=SD          grid=(64, 48)         copy/req= 25.88 MiB  avg= 1654.98 us  BW~2098.4 GB/s
reqs=128_conv=SD         grid=(128, 48)        copy/req= 25.88 MiB  avg= 2688.44 us  BW~2583.6 GB/s


PR (postprocess_vectorize)

device: NVIDIA GB200
model: Qwen/Qwen3.5-9B tp=1 num_spec=2
linear_layers=24 state_types=2 total_states=48
conv: dim=8192 state_len=5 (=80.0 KiB/block)
temporal: inner=524288 (=1.00 MiB/block)
iters warmup=20 timed=200

reqs=1_conv=SD           grid=(1, 48)          copy/req= 25.88 MiB  avg=   63.82 us  BW~ 850.2 GB/s
reqs=4_conv=SD           grid=(4, 48)          copy/req= 25.88 MiB  avg=  129.82 us  BW~1671.9 GB/s
reqs=8_conv=SD           grid=(8, 48)          copy/req= 25.88 MiB  avg=  141.37 us  BW~3070.8 GB/s
reqs=16_conv=SD          grid=(16, 48)         copy/req= 25.88 MiB  avg=  184.14 us  BW~4714.9 GB/s
reqs=32_conv=SD          grid=(32, 48)         copy/req= 25.88 MiB  avg=  298.45 us  BW~5818.1 GB/s
reqs=64_conv=SD          grid=(64, 48)         copy/req= 25.88 MiB  avg=  631.86 us  BW~5496.3 GB/s
reqs=128_conv=SD         grid=(128, 48)        copy/req= 25.88 MiB  avg= 1153.36 us  BW~6022.2 GB/s

