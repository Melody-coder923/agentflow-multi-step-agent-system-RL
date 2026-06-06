# AgentFlow: Multi-Step Tool-Use Agent

A multi-step agent that decomposes a question, calls tools, verifies intermediate results, and writes a final answer. A single Qwen3.5-0.8B serves all four roles via prompt templating, fine-tuned with LoRA + GRPO. Built on the AgentFlow framework (arXiv:2510.05592).

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

There is one model. The trained Qwen3.5-0.8B (with LoRA adapter) is invoked four times per turn using four different prompt templates — one per role. `MODEL_ENGINE="trainable,trainable,trainable,trainable"` in the eval scripts confirms this: no external LLM is in the inference path.

Tools enabled during the in-domain benchmarks (`test/run_lora_bench.sh`):

- `Base_Generator_Tool` — direct LLM fallback
- `Google_Search_Tool`
- `Wikipedia_Search_Tool`

An `Execute_SQL_Tool` (`AgentFlow/agentflow/tools/execute_sql_tool.py`) is added at Spider eval time only; the trained model never sees it during training.

Memory is a per-question Python dict keyed by step number, storing `(tool_name, sub_goal, command, result)` per step. It is stringified into Planner/Verifier prompts via `repr()` and discarded when the question is done. No vector store, no compression.

## Training

`train/modal_train_agent.py` runs the production training on Modal L40S using TRL's `GRPOTrainer` with PEFT-LoRA.

| | |
|---|---|
| Base model | `Qwen/Qwen3.5-0.8B` |
| Method | LoRA + GRPO (single-turn completion) |
| LoRA | r=8, α=32, targets q/k/v/o + gate/up/down |
| Optimizer | lr=1e-5, bf16 |
| Batch | per-device 4, grad accum 4 (effective 16) |
| Max steps | 3000 |
| Generations per prompt | 8 |
| Data | `data/train/combined_train.parquet` (mixed search QA + math hard) |
| Saved to | `results/final_qwen35_lora` |

Reward function (`reward_logic_function`):

```
+0.2 each for <think>, </think>, <answer>, </answer>      (format, max 0.8)
+2.0 if extracted <answer> content == groundtruth         (EM, lowercased)
+0.5 if groundtruth string appears anywhere in completion (substring)
```

Range 0 – 2.8. GRPO turns these scalar rewards into group-relative advantages over the 8 generations and applies a clipped PPO-style update with KL against the reference model.

Training is single-turn: the model is fed a question prompt and generates a flat `<think>...</think><answer>...</answer>` completion. The 4-role loop is only assembled at inference time, and the same trained model handles all four roles. This is not the in-the-flow training described in the original paper.

The repo also contains a verl-based multi-turn pipeline (`train/train_agent.py` → `agentflow.verl` → `train/rollout.py`) that drives the full loop during training and uses gpt-4o as an LLM-judge for the reward (`train/utils.py::compute_score`). This is not what produced the released checkpoint — `serve_lora_local.py` loads `results/final_qwen35_lora`, which is the save path of the Modal/TRL script.

## Serving

Two paths are implemented:

- `serve_lora_local.py` — FastAPI + transformers + PEFT on a local GPU (H200 in our setup). Default adapter: `Skypioneer/qwen35-0.8b-agentflow-lora` on HF Hub.
- `modal_serve.py` — Modal-hosted backend. Uses SGLang for Qwen3.5 (vLLM has a weight-prefix incompatibility with Qwen3.5 at the time of writing) and vLLM for non-Qwen3.5 baselines.

Inference defaults (`AgentFlow/agentflow/solver.py`): `max_steps=10`, `max_time=300s`.

For SLURM/H200 deployment see `run_lora_h200.sbatch` and `cluster_check.sh`.

## Results

### In-domain (5 multi-hop benchmarks)

Baseline is the same Qwen3.5-0.8B without LoRA, running through the identical loop with the same three tools and the same eval. The only variable is the LoRA adapter.

| Benchmark | No-Training | LoRA-Trained | Δ |
|---|---:|---:|---:|
| HotpotQA | 8.0 | 30.0 | +22.0 |
| 2Wiki | 15.0 | 28.0 | +13.0 |
| Bamboogle | 10.4 | 18.0 | +7.6 |
| GAIA | 0.0 | 6.0 | +6.0 |
| MuSiQue | 3.0 | 6.0 | +3.0 |

Eval scoring uses gpt-4o as a semantic-equivalence judge (`test/calculate_score_unified.py::ResultScorer`), not exact match.

### Cross-task transfer: Text-to-SQL

The model never sees an SQL tool during training. At test time an `Execute_SQL_Tool` description is added to the toolbox and the system is evaluated on Spider with execution accuracy:

| Model | Spider EX |
|---|---:|
| Qwen2.5-7B-Instruct, zero-shot | 35 |
| Qwen3.5-0.8B + LoRA (this work) | 50 |

The same 0.8B selects the SQL tool (as Planner) and writes the SQL string (as Executor). No SQL-specific fine-tuning.

## Limitations

- Absolute scores are below leaderboard SOTA. The numbers above measure the lift over the same untrained model, not a comparison with SOTA systems.
- Training reward is rule-based (format + EM + substring); eval uses an LLM-judge. The mismatch can underweight semantically-correct-but-paraphrased answers during training.
- Training is single-turn; the multi-turn loop is only at inference. The model does not realize the in-the-flow credit assignment described in the original paper.
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
modal_serve.py          Modal serving (SGLang for Qwen3.5, vLLM otherwise)
run_lora_h200.sbatch    SLURM deployment

test/
  run_lora_bench.sh     run a benchmark against the LoRA planner server
  {bamboogle,2wiki,hotpotqa,musique,gaia}/   per-benchmark scripts
  text2sql/             Spider eval
  calculate_score_unified.py  gpt-4o LLM-judge scorer
```

## Quick start

```bash
pip install -r requirements_lora_serve.txt
python serve_lora_local.py                        # serve adapter on a local GPU
cd test && bash run_lora_bench.sh bamboogle       # run a benchmark
```

For training: `python train/modal_train_agent.py` (requires a Modal account).

## Credits

Built on AgentFlow: *In-the-Flow Agentic System Optimization* (arXiv:2510.05592).

Additions in this repo: single-turn LoRA + GRPO training on Qwen3.5-0.8B via TRL + Modal, FastAPI + transformers serving on H200, an SGLang-based Modal serving fallback for Qwen3.5, and a zero-shot Text-to-SQL transfer evaluation.
