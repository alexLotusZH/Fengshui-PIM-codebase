#  Operator Level Test
cd "$(dirname "$0")"


operators=(
    "trace_qkv_proj"
    "trace_attn_score"
    "trace_attn_softmax"
    "trace_attn_o"
    "trace_wo_proj"
    "trace_w1_proj"
    "trace_w3_proj"
    "trace_w2_proj"
)

# rm cent_simulation/processed_*.csv

for OP in "${operators[@]}"
do

echo "========================================"
echo "Starting Test Iter on: $OP"
echo "========================================"

rm -rf trace/32_channels_per_device/model_parallel/Llama31-8B/trace_*.txt*
rm -rf trace/32_channels_per_device/model_parallel/Llama31-8B/logs.txt
rm -rf trace/32_channels_per_device/model_parallel/Llama31-8B/result.txt
rm -rf trace/32_channels_per_device/model_parallel/Llama31-8B/compiled_results.txt

rm -rf trace/32_channels_per_device/model_parallel_embedding/Llama31-8B/trace_*.txt*
rm -rf trace/32_channels_per_device/model_parallel_embedding/Llama31-8B/logs.txt
rm -rf trace/32_channels_per_device/model_parallel_embedding/Llama31-8B/result.txt
rm -rf trace/32_channels_per_device/model_parallel_embedding/Llama31-8B/compiled_results.txt

rm -rf trace/32_channels_per_device/model_parallel_FC/Llama31-8B/trace_*.txt*
rm -rf trace/32_channels_per_device/model_parallel_FC/Llama31-8B/logs.txt
rm -rf trace/32_channels_per_device/model_parallel_FC/Llama31-8B/result.txt
rm -rf trace/32_channels_per_device/model_parallel_FC/Llama31-8B/compiled_results.txt

rm -rf trace/32_channels_per_device/pipeline_parallel/Llama31-8B/trace_*.txt*
rm -rf trace/32_channels_per_device/pipeline_parallel/Llama31-8B/logs.txt
rm -rf trace/32_channels_per_device/pipeline_parallel/Llama31-8B/result.txt
rm -rf trace/32_channels_per_device/pipeline_parallel/Llama31-8B/compiled_results.txt

rm -rf trace/32_channels_per_device/pipeline_parallel_embedding/Llama31-8B/trace_*.txt*
rm -rf trace/32_channels_per_device/pipeline_parallel_embedding/Llama31-8B/logs.txt
rm -rf trace/32_channels_per_device/pipeline_parallel_embedding/Llama31-8B/result.txt
rm -rf trace/32_channels_per_device/pipeline_parallel_embedding/Llama31-8B/compiled_results.txt


rm -rf trace/32_channels_per_device/model_parallel/Llama31-70B/trace_*.txt*
rm -rf trace/32_channels_per_device/model_parallel/Llama31-70B/logs.txt
rm -rf trace/32_channels_per_device/model_parallel/Llama31-70B/result.txt
rm -rf trace/32_channels_per_device/model_parallel/Llama31-70B/compiled_results.txt

rm -rf trace/32_channels_per_device/model_parallel_embedding/Llama31-70B/trace_*.txt*
rm -rf trace/32_channels_per_device/model_parallel_embedding/Llama31-70B/logs.txt
rm -rf trace/32_channels_per_device/model_parallel_embedding/Llama31-70B/result.txt
rm -rf trace/32_channels_per_device/model_parallel_embedding/Llama31-70B/compiled_results.txt

rm -rf trace/32_channels_per_device/model_parallel_FC/Llama31-70B/trace_*.txt*
rm -rf trace/32_channels_per_device/model_parallel_FC/Llama31-70B/logs.txt
rm -rf trace/32_channels_per_device/model_parallel_FC/Llama31-70B/result.txt
rm -rf trace/32_channels_per_device/model_parallel_FC/Llama31-70B/compiled_results.txt

rm -rf trace/32_channels_per_device/pipeline_parallel/Llama31-70B/trace_*.txt*
rm -rf trace/32_channels_per_device/pipeline_parallel/Llama31-70B/logs.txt
rm -rf trace/32_channels_per_device/pipeline_parallel/Llama31-70B/result.txt
rm -rf trace/32_channels_per_device/pipeline_parallel/Llama31-70B/compiled_results.txt

rm -rf trace/32_channels_per_device/pipeline_parallel_embedding/Llama31-70B/trace_*.txt*
rm -rf trace/32_channels_per_device/pipeline_parallel_embedding/Llama31-70B/logs.txt
rm -rf trace/32_channels_per_device/pipeline_parallel_embedding/Llama31-70B/result.txt
rm -rf trace/32_channels_per_device/pipeline_parallel_embedding/Llama31-70B/compiled_results.txt


bash remove_old_results.sh
cd cent_simulation
# bash simulation.sh <set threads based on your platform> <sequence length gap>
# bash simulation.sh 128 1024

model="Llama31-8B"
# run prefill stage only
for idx in 512 1024 2048 4096
do
rm simulation_results.csv

python3 run_sim.py --model $model --operator $OP  --num_devices 1 --model_parallel --generate_trace --simulate_trace --process_results --update_csv --run_simulation_max_workers 256 --generate_trace_max_workers 256 --seqlen_gap 256 --prefill $idx --decoding 0
python3 run_sim.py --model $model --operator $OP  --num_devices 2 --model_parallel --generate_trace --simulate_trace --process_results --update_csv --run_simulation_max_workers 256 --generate_trace_max_workers 256 --seqlen_gap 256 --prefill $idx --decoding 0

python3 run_sim.py --model $model  --operator $OP --model_parallel --process_throughputs --num_devices 1 --phase prefill --prefill $idx --decoding 0 --simulation_result_path simulation_results.csv --processed_result_path processed_results_${model}_${OP}_${idx}.csv
python3 run_sim.py --model $model  --operator $OP --model_parallel --process_throughputs --num_devices 2 --phase prefill --prefill $idx --decoding 0 --simulation_result_path simulation_results.csv --processed_result_path processed_results_${model}_${OP}_${idx}.csv
done

# run decoding stage only
# python3 run_sim.py --model Llama31-70B --operator $OP  --num_devices 1 --model_parallel --generate_trace --simulate_trace --process_results --update_csv --run_simulation_max_workers 128 --generate_trace_max_workers 128 --seqlen 512 1024 2048 4096
# python3 run_sim.py --model Llama31-70B --operator $OP  --num_devices 2 --model_parallel --generate_trace --simulate_trace --process_results --update_csv --run_simulation_max_workers 128 --generate_trace_max_workers 128 --seqlen 512 1024 2048 4096

cd ..

done



# bash generate_figures.sh

