# Operator-Level Profiling Extension

Artifact of the MICRO 2026 paper *Fengshui: Demystifying Chiplet Ecosystem and Bespoke Neural Network Accelerator Codesign*, built on the CENT artifact (ASPLOS 2025).

## New Features

1. **Operator-level profiling** — simulate the time & power of each transformer operator individually instead of the whole block.
2. **New models** — now supports Llama3.1-8B/70B, Qwen3-30B-A3B / 235B-A22B (MoE), OPT-66B, and ViT-B16/L16/H14.
3. **Power-calculation fixes** — small bug fixes; accelerator (RMSNorm/Softmax/RoPE) latency & energy are now only added when the corresponding operator is simulated.

## Modified Files

- `GPT.py` / `Llama.py` — the monolithic `trace_only()` transformer-block trace is split into per-operator trace methods (`trace_rms`, `trace_qkv_proj`, `trace_rope`, `trace_attn_score`, `trace_attn_softmax`, `trace_attn_o`, `trace_wo_proj`, `trace_router`, `trace_w1_proj`, `trace_w3_proj`, `trace_ffn_af`, `trace_w2_proj`).
- `function_sim.py` — new `--operator` argument dispatches to a single operator trace (`all` reproduces the original full-block behavior); Llama models now use `Llama3_1.py` (adds GQA/MoE support).
- `cent_power_calculator.py` — `power_calculator()` takes an `Operator` argument and gates RMSNorm/Softmax/RoPE latency & energy terms by the selected operator; power is normalized by total latency (PIM + accelerator); prints per-operator latency breakdown as CSV (`pim, RMS, SFT, ROT, Total Acc, Total, utilization`).
- `utils.py` — model configs for the new models and the `--operator` CLI option.

## Quick Usage

Profile a single operator on an end2end workflow with a default seq_length 4096:

```bash
python3 run_sim.py --model Llama31-8B --operator trace_qkv_proj --num_devices 1 --model_parallel \
--generate_trace --simulate_trace --process_results --update_csv --process_throughputs  --processed_result_path processed_results_Llama31-8B_trace_qkv_proj_end2end.csv
```
Raw results at different sequence length are written to `simulation_results.csv` under the current dir, while the processed and averaged result of the total process is written to `processed_results_<model>_<operator>_<seqlen>.csv` under the current dir. 
 
Try different configurations of PIM:

Refer to `utils.py` and specify args when running `run_sim.py`

## Detailed Usage

`run_sim.py` arguments:

| Argument | Default | Description |
|---|---|---|
| `--model` | *(needs specification)* | Model to simulate (Llama2-7B/13B/70B, OPT-66B, Llama31-8B/70B, Qwen3-30B-A3B/235B-A22B, ViT-B16/H14/L16). |
| `--operator` | `all` | Operator to profile (e.g. `trace_qkv_proj`, `trace_attn_score`); `all` for the full transformer block. |
| `--num_devices` | `32` | Number of CXL devices. |
| `--num_channels` | `32` | Number of PIM channels per device. |
| `--PCIE_lanes` | `144` | Total number of PCIe lanes, split evenly across devices. |
| `--reuse_size` | `32` | Global-buffer reuse size, which depends on the register count. |
| `--model_parallel` | off | Use tensor + pipeline parallelism and sweep all TP/PP splits of `--num_devices`; without it, pipeline parallelism only. |
| `--inter-device-attention` | off | Split attention across devices; applies only with `--model_parallel`. |
| `--prefill` | `512` | Prefill length. |
| `--decoding` | `3584` | Decoding length. |
| `--seqlen` | — | Explicit list of sequence lengths to simulate; overrides `--seqlen_gap`. |
| `--seqlen_gap` | `128` | Step between simulated sequence lengths, up to `prefill + decoding`. |
| `--phase` | `end2end` | Phase averaged by `--process_throughputs`: `prefill`, `decoding`, or `end2end`. |
| `--generate_trace` | off | Generate PIM traces under `../trace/`. |
| `--simulate_trace` | off | Run the traces through Ramulator. |
| `--process_results` | off | Compile the Ramulator logs into latency results. |
| `--update_csv` | off | Write per-seqlen latency, energy, and power to `--simulation_result_path`. |
| `--process_throughputs` | off | Average results over `--phase` and write them to `--processed_result_path`. |
| `--simulation_result_path` | `simulation_results.csv` | Output CSV for per-seqlen results. |
| `--processed_result_path` | `processed_results.csv` | Output CSV for phase-averaged results. |
| `--generate_trace_max_workers` | `20` | Maximum number of parallel trace-generation jobs. |
| `--run_simulation_max_workers` | `4` | Maximum number of parallel Ramulator jobs. |



## Test Scripts Examples (repo root)

| Script | Purpose |
|---|---|
| `test1.sh` | Full-block baseline (Llama2-7B/13B/70B) |
| `test2.sh` | Prefill vs decode (Llama2-13B) |
| `test3.sh` | Operator-level, Llama31-8B |
| `test4.sh` / `test5.sh` | OPT-66B prefill vs decode |
| `test6.sh` / `test7.sh` | Operator-level, Qwen3-235B-A22B / Qwen3-30B-A3B |
| `test8.sh` / `test9.sh` / `test10.sh` | Operator-level, ViT-B16 / ViT-H14 / ViT-L16 |
| `test_prefill.sh` | Clean processed CSVs and rerun test8 |

All PIM-related data used in "Fengshui" is generated in this codebase by these scripts. Each script cleans old traces, then loops `run_sim.py` over the operator list at various device counts.
