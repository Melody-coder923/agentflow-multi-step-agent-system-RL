# AgentFlow: Multi-Step Agent for Tool-Based Reasoning

A multi-step agent that decomposes a question, calls tools, verifies intermediate results, and produces a final answer. The Planner is fine-tuned with LoRA + GRPO on Qwen3.5-0.8B. Built on the AgentFlow framework (arXiv:2510.05592).

## Architecture

The inference loop has four roles:

```
question → Planner → Executor → Memory → Verifier ──┐
              ▲                                      │
              └────── continue ──────────────────────┘
                                                     │ STOP
                                                     ▼
                                                  Generator → answer
```

Only the Planner's `generate_next_step` is trainable. Query analysis and final-answer generation live in the same `Planner` class but call a frozen `llm_engine_fixed` (gpt-4o-mini). Executor and Verifier are gpt-4o-mini as well.

Tools enabled during training:

- `Base_Generator_Tool` — direct LLM fallback
- `Python_Coder_Tool`
- `Google_Search_Tool`
- `Wikipedia_Search_Tool`

An SQL execution tool (`AgentFlow/agentflow/tools/execute_sql_tool.py`) is added at Spider eval time only — the trained Planner has never seen it.

Memory is a per-question Python dict keyed by step number, storing `(tool_name, sub_goal, command, result)` per step. It is stringified into Planner/Verifier prompts via `repr()` and discarded when the question is done. No vector store, no compression.

## Training

`train/modal_train_agent.py` runs the production training on Modal L40S.

| | |
|---|---|
| Base model | `Qwen/Qwen3.5-0.8B` |
| Method | LoRA + GRPO (single-turn) |
| LoRA | r=8, α=32, targets q/k/v/o + gate/up/down |
| Optimizer | lr=1e-5, bf16 |
| Batch | per-device 4, grad accum 4 (effective 16) |
| Max steps | 3000 |
| Generations per prompt | 8 |
| Data | `data/train/combined_train.parquet` |
| Saved to | `results/final_qwen35_lora` |

The reward function (`reward_logic_function` in the same file):

```
+0.2 each for <think>, </think>, <answer>, </answer>      (format, max 0.8)
+2.0 if extracted <answer> content == groundtruth         (EM, lowercased)
+0.5 if groundtruth string appears anywhere in completion (substring)
```

Range 0 – 2.8. GRPO turns this into group-relative advantages over the 8 generations and applies a clipped PPO-style update with KL against the reference model.

Training is single-turn: the model produces flat `<think>...</think><answer>...</answer>` completions. The 4-role loop is only assembled at inference time. This differs from the in-the-flow training in the original paper.

The repo also contains a verl-based multi-turn pipeline (`train/train_agent.py` → `agentflow.verl` → `train/rollout.py`) that drives the full loop during training and uses gpt-4o as an LLM-judge for the reward (`train/utils.py::compute_score`). This pipeline is not what produced the released checkpoint — `serve_lora_local.py` loads `results/final_qwen35_lora`, which is the save path of the Modal/TRL script.

## Serving

```bash
python serve_lora_local.py
```

Loads the trained LoRA adapter and exposes the Planner as an OpenAI-compatible endpoint. Default adapter: `Skypioneer/qwen35-0.8b-agentflow-lora` on HF Hub. Override with `MERGED_MODEL` or `LORA_DIR` env vars.

Inference defaults (`AgentFlow/agentflow/solver.py`): `max_steps=10`, `max_time=300s`, frozen modules via OpenAI gpt-4o-mini.

For SLURM/H200 deployment see `run_lora_h200.sbatch` and `cluster_check.sh`.

## Results

Five multi-hop benchmarks. Baseline is the same Qwen3.5-0.8B Planner without LoRA, plugged into the identical inference loop with identical frozen modules and tools.

| Benchmark | No-Training | LoRA-Trained | Δ |
|---|---:|---:|---:|
| HotpotQA | 8.0 | 30.0 | +22.0 |
| 2Wiki | 15.0 | 28.0 | +13.0 |
| Bamboogle | 10.4 | 18.0 | +7.6 |
| GAIA | 0.0 | 6.0 | +6.0 |
| MuSiQue | 3.0 | 6.0 | +3.0 |

Eval scoring uses gpt-4o as a semantic-equivalence judge (`test/calculate_score_unified.py::ResultScorer`), not exact match.

### Cross-task transfer: Text-to-SQL

The trained Planner never sees an SQL tool. At test time `Execute_SQL_Tool` is added to the Executor's toolbox and the system is evaluated on Spider with execution accuracy:

| Model | Spider EX |
|---|---:|
| Qwen2.5-7B-Instruct, zero-shot | 35 |
| Qwen3.5-0.8B + LoRA (this work) | 50 |

Zero-shot tool acquisition — no SQL-specific fine-tuning.

## Limitations

- Absolute scores are below leaderboard SOTA. The numbers above measure the lift over the same untrained planner, not a comparison with SOTA systems.
- Training reward is rule-based (format + EM + substring); eval uses LLM-judge. The mismatch can underweight semantically-correct-but-paraphrased answers during training.
- Training is single-turn; the multi-turn loop is only at inference. The Planner does not realize the in-the-flow credit assignment described in the original paper.
- `max_steps=10` at inference. GAIA and MuSiQue often need longer chains.

## Repository layout

```
AgentFlow/agentflow/
  solver.py             main inference loop
  models/               planner, executor, verifier, memory
  tools/                tool implementations (incl. execute_sql_tool.py)

train/
  modal_train_agent.py  production: TRL GRPOTrainer on Modal L40S
  train_agent.py        alt: verl entrypoint
  rollout.py            alt: multi-turn rollout with LLM-judge reward
  utils.py              compute_score (gpt-4o judge, used by alt pipeline)
  config.yaml           verl config

serve_lora_local.py     FastAPI serving on H200
run_lora_h200.sbatch    SLURM deployment
test/                   one directory per benchmark
```

## Quick start

```bash
pip install -r requirements_lora_serve.txt
python quick_start.py            # smoke test
python serve_lora_local.py       # serve adapter
cd test/bamboogle && bash run.sh # run a benchmark
```

For training: `python train/modal_train_agent.py` (requires a Modal account).

## Credits

Built on AgentFlow: *In-the-Flow Agentic System Optimization* (arXiv:2510.05592).

Additions in this repo: single-turn LoRA + GRPO training on Qwen3.5-0.8B via TRL + Modal, FastAPI/SLURM serving on H200, and a zero-shot Text-to-SQL transfer evaluation.
