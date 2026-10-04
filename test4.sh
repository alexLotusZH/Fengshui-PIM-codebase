#  OPT prefill vs decoding
cd "$(dirname "$0")"

rm -rf trace/32_channels_per_device/model_parallel/OPT-66B/trace_*.txt*
rm -rf trace/32_channels_per_device/model_parallel/OPT-66B/logs.txt
rm -rf trace/32_channels_per_device/model_parallel/OPT-66B/result.txt
rm -rf trace/32_channels_per_device/model_parallel/OPT-66B/compiled_results.txt

rm -rf trace/32_channels_per_device/model_parallel_embedding/OPT-66B/trace_*.txt*
rm -rf trace/32_channels_per_device/model_parallel_embedding/OPT-66B/logs.txt
rm -rf trace/32_channels_per_device/model_parallel_embedding/OPT-66B/result.txt
rm -rf trace/32_channels_per_device/model_parallel_embedding/OPT-66B/compiled_results.txt

rm -rf trace/32_channels_per_device/model_parallel_FC/OPT-66B/trace_*.txt*
rm -rf trace/32_channels_per_device/model_parallel_FC/OPT-66B/logs.txt
rm -rf trace/32_channels_per_device/model_parallel_FC/OPT-66B/result.txt
rm -rf trace/32_channels_per_device/model_parallel_FC/OPT-66B/compiled_results.txt

rm -rf trace/32_channels_per_device/pipeline_parallel/OPT-66B/trace_*.txt*
rm -rf trace/32_channels_per_device/pipeline_parallel/OPT-66B/logs.txt
rm -rf trace/32_channels_per_device/pipeline_parallel/OPT-66B/result.txt
rm -rf trace/32_channels_per_device/pipeline_parallel/OPT-66B/compiled_results.txt

rm -rf trace/32_channels_per_device/pipeline_parallel_embedding/OPT-66B/trace_*.txt*
rm -rf trace/32_channels_per_device/pipeline_parallel_embedding/OPT-66B/logs.txt
rm -rf trace/32_channels_per_device/pipeline_parallel_embedding/OPT-66B/result.txt
rm -rf trace/32_channels_per_device/pipeline_parallel_embedding/OPT-66B/compiled_results.txt



bash remove_old_results.sh
cd cent_simulation
# bash simulation.sh <set threads based on your platform> <sequence length gap>
# bash simulation.sh 64 1024

# prefill Llama2
python3 run_sim.py --model OPT-66B --num_devices 1 --model_parallel --generate_trace --simulate_trace --process_results --update_csv --run_simulation_max_workers 64 --generate_trace_max_workers 64 --prefill 512 --decoding 0 --phase prefill --seqlen_gap 256

# decode Llama2
python3 run_sim.py --model OPT-66B --num_devices 1 --model_parallel --generate_trace --simulate_trace --process_results --update_csv --run_simulation_max_workers 64 --generate_trace_max_workers 64 --prefill 0 --phase decoding --seqlen_gap 1 --seqlen_gap 1024

# process data prefill
python3 run_sim.py --model OPT-66B --model_parallel --process_throughputs --num_devices 1 --phase prefill --simulation_result_path simulation_results.csv --prefill 512 --decoding 0

# python3 run_sim.py --model OPT-66B --model_parallel --process_throughputs --num_devices 32 --phase $phase --simulation_result_path simulation_results.csv


cd ..
# bash generate_figures.sh

