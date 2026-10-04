#  Operator Level Test
cd "$(dirname "$0")"

rm -rf trace/32_channels_per_device/model_parallel/Llama2-70B/trace_*.txt*
rm -rf trace/32_channels_per_device/model_parallel/Llama2-70B/logs.txt
rm -rf trace/32_channels_per_device/model_parallel/Llama2-70B/result.txt
rm -rf trace/32_channels_per_device/model_parallel/Llama2-70B/compiled_results.txt
rm -rf trace/32_channels_per_device/model_parallel_embedding/Llama2-70B/trace_*.txt*
rm -rf trace/32_channels_per_device/model_parallel_embedding/Llama2-70B/logs.txt
rm -rf trace/32_channels_per_device/model_parallel_embedding/Llama2-70B/result.txt
rm -rf trace/32_channels_per_device/model_parallel_embedding/Llama2-70B/compiled_results.txt
rm -rf trace/32_channels_per_device/model_parallel_FC/Llama2-70B/trace_*.txt*
rm -rf trace/32_channels_per_device/model_parallel_FC/Llama2-70B/logs.txt
rm -rf trace/32_channels_per_device/model_parallel_FC/Llama2-70B/result.txt
rm -rf trace/32_channels_per_device/model_parallel_FC/Llama2-70B/compiled_results.txt
rm -rf trace/32_channels_per_device/pipeline_parallel/Llama2-70B/trace_*.txt*
rm -rf trace/32_channels_per_device/pipeline_parallel/Llama2-70B/logs.txt
rm -rf trace/32_channels_per_device/pipeline_parallel/Llama2-70B/result.txt
rm -rf trace/32_channels_per_device/pipeline_parallel/Llama2-70B/compiled_results.txt
rm -rf trace/32_channels_per_device/pipeline_parallel_embedding/Llama2-70B/trace_*.txt*
rm -rf trace/32_channels_per_device/pipeline_parallel_embedding/Llama2-70B/logs.txt
rm -rf trace/32_channels_per_device/pipeline_parallel_embedding/Llama2-70B/result.txt
rm -rf trace/32_channels_per_device/pipeline_parallel_embedding/Llama2-70B/compiled_results.txt

rm -rf trace/32_channels_per_device/model_parallel/Llama2-7B/trace_*.txt*
rm -rf trace/32_channels_per_device/model_parallel/Llama2-7B/logs.txt
rm -rf trace/32_channels_per_device/model_parallel/Llama2-7B/result.txt
rm -rf trace/32_channels_per_device/model_parallel/Llama2-7B/compiled_results.txt

rm -rf trace/32_channels_per_device/model_parallel_embedding/Llama2-7B/trace_*.txt*
rm -rf trace/32_channels_per_device/model_parallel_embedding/Llama2-7B/logs.txt
rm -rf trace/32_channels_per_device/model_parallel_embedding/Llama2-7B/result.txt
rm -rf trace/32_channels_per_device/model_parallel_embedding/Llama2-7B/compiled_results.txt

rm -rf trace/32_channels_per_device/model_parallel_FC/Llama2-7B/trace_*.txt*
rm -rf trace/32_channels_per_device/model_parallel_FC/Llama2-7B/logs.txt
rm -rf trace/32_channels_per_device/model_parallel_FC/Llama2-7B/result.txt
rm -rf trace/32_channels_per_device/model_parallel_FC/Llama2-7B/compiled_results.txt

rm -rf trace/32_channels_per_device/pipeline_parallel/Llama2-7B/trace_*.txt*
rm -rf trace/32_channels_per_device/pipeline_parallel/Llama2-7B/logs.txt
rm -rf trace/32_channels_per_device/pipeline_parallel/Llama2-7B/result.txt
rm -rf trace/32_channels_per_device/pipeline_parallel/Llama2-7B/compiled_results.txt

rm -rf trace/32_channels_per_device/pipeline_parallel_embedding/Llama2-7B/trace_*.txt*
rm -rf trace/32_channels_per_device/pipeline_parallel_embedding/Llama2-7B/logs.txt
rm -rf trace/32_channels_per_device/pipeline_parallel_embedding/Llama2-7B/result.txt
rm -rf trace/32_channels_per_device/pipeline_parallel_embedding/Llama2-7B/compiled_results.txt


rm -rf trace/32_channels_per_device/model_parallel/Llama2-13B/trace_*.txt*
rm -rf trace/32_channels_per_device/model_parallel/Llama2-13B/logs.txt
rm -rf trace/32_channels_per_device/model_parallel/Llama2-13B/result.txt
rm -rf trace/32_channels_per_device/model_parallel/Llama2-13B/compiled_results.txt

rm -rf trace/32_channels_per_device/model_parallel_embedding/Llama2-13B/trace_*.txt*
rm -rf trace/32_channels_per_device/model_parallel_embedding/Llama2-13B/logs.txt
rm -rf trace/32_channels_per_device/model_parallel_embedding/Llama2-13B/result.txt
rm -rf trace/32_channels_per_device/model_parallel_embedding/Llama2-13B/compiled_results.txt

rm -rf trace/32_channels_per_device/model_parallel_FC/Llama2-13B/trace_*.txt*
rm -rf trace/32_channels_per_device/model_parallel_FC/Llama2-13B/logs.txt
rm -rf trace/32_channels_per_device/model_parallel_FC/Llama2-13B/result.txt
rm -rf trace/32_channels_per_device/model_parallel_FC/Llama2-13B/compiled_results.txt

rm -rf trace/32_channels_per_device/pipeline_parallel/Llama2-13B/trace_*.txt*
rm -rf trace/32_channels_per_device/pipeline_parallel/Llama2-13B/logs.txt
rm -rf trace/32_channels_per_device/pipeline_parallel/Llama2-13B/result.txt
rm -rf trace/32_channels_per_device/pipeline_parallel/Llama2-13B/compiled_results.txt

rm -rf trace/32_channels_per_device/pipeline_parallel_embedding/Llama2-13B/trace_*.txt*
rm -rf trace/32_channels_per_device/pipeline_parallel_embedding/Llama2-13B/logs.txt
rm -rf trace/32_channels_per_device/pipeline_parallel_embedding/Llama2-13B/result.txt
rm -rf trace/32_channels_per_device/pipeline_parallel_embedding/Llama2-13B/compiled_results.txt


bash remove_old_results.sh
cd cent_simulation
# bash simulation.sh <set threads based on your platform> <sequence length gap>
# bash simulation.sh 128 1024

python3 run_sim.py --model Llama2-7B --num_devices 1 --model_parallel --generate_trace --simulate_trace --process_results --update_csv --run_simulation_max_workers 128 --generate_trace_max_workers 128 --seqlen_gap 1024
python3 run_sim.py --model Llama2-13B --num_devices 1 --model_parallel --generate_trace --simulate_trace --process_results --update_csv --run_simulation_max_workers 128 --generate_trace_max_workers 128 --seqlen_gap 1024
python3 run_sim.py --model Llama2-70B --num_devices 1 --model_parallel --generate_trace --simulate_trace --process_results --update_csv --run_simulation_max_workers 128 --generate_trace_max_workers 128 --seqlen_gap 1024

# python3 run_sim.py --model Llama2-70B --model_parallel --process_throughputs --num_devices 32 --phase $phase --simulation_result_path simulation_results.csv


cd ..
# bash generate_figures.sh

