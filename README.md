# AgentFlow: Multi-Step Agent System for Tool-Based Reasoning and Execution

A multi-step agent framework that plans tool use, executes actions, verifies outcomes, and improves decision quality through Flow-GRPO reinforcement learning.

---

## 1. Overview

This project implements an autonomous agent system that solves complex tasks through iterative reasoning and tool use.

Instead of generating answers in a single step, the system follows a loop:

```
plan → act → verify → refine
```

enabling more reliable and interpretable behavior.

It supports both:

- open-ended reasoning tasks (e.g., multi-hop QA)
- structured data tasks (e.g., Text-to-SQL)

---

## 2. Architecture

The system is built around a **Planner–Executor–Verifier** loop:

- **Planner** — decomposes the task and selects tools
- **Executor** — executes actions using tools (search, SQL, etc.)
- **Verifier** — evaluates intermediate results and decides whether to stop, retry, or continue

```
       ┌───────────┐       ┌───────────┐       ┌───────────┐
user → │  Planner  │ ────▶ │  Executor │ ────▶ │  Verifier │
       └───────────┘       └───────────┘       └─────┬─────┘
             ▲                                       │
             └─────────── replan on failure ─────────┘
```

This creates an iterative decision loop rather than a one-shot response system.

---

## 3. Key Capabilities

- **Multi-step reasoning** — handles complex tasks through iterative planning instead of one-shot generation
- **Tool-based execution** — integrates external tools such as search and SQL for grounded reasoning
- **Hybrid verification** — combines LLM-based semantic judgment with rule-based execution checks
- **Cross-task generalization** — extends to new tasks (e.g., Text-to-SQL) without task-specific retraining
- **RL-based decision improvement** — improves planner behavior through Flow-GRPO + LoRA training

---

## 4. Design Decisions

**Planner–Executor–Verifier instead of one-shot generation.**
Separating decision-making, execution, and validation improves reliability on multi-step tasks and makes failures localizable.

**Hybrid verification strategy.**
LLM-based judgment is used for open-ended outputs (QA benchmarks); execution-based checks are used for structured outputs (SQL execution match). Code-level error handling catches hard tool failures before they reach the model.

**Role-specific model selection.**
The Planner is the trainable component (Qwen3.5-0.8B with LoRA). The Verifier and Executor use a fixed external model (gpt-4o-mini). This lets us optimize planning behavior in isolation without retraining the rest of the stack.

**LoRA for efficient adaptation.**
LoRA (rank 8, <1% trainable parameters) enables low-cost fine-tuning of planner behavior without full model retraining — critical for iterating on RL reward design.

**Flow-GRPO for planner optimization.**
GRPO's group-relative advantage estimation is well-suited to trajectory-level rewards in tool-using agents, and removes the need for a separate value network.

---

## 5. Results

Evaluated across 5 multi-hop reasoning benchmarks on a single H200 GPU:

| Benchmark | No-Training | LoRA-Training | Delta |
|---|---:|---:|---:|
| HotpotQA | 8.0 | 30.0 | **+22.0** |
| 2Wiki | 15.0 | 28.0 | +13.0 |
| Bamboogle | 10.4 | 18.0 | +7.6 |
| GAIA | 0.0 | 6.0 | +6.0 |
| MuSiQue | 3.0 | 6.0 | +3.0 |

LoRA training improves performance across **all five tasks**, with the strongest gain on multi-hop QA (HotpotQA: 8% → 30%).

Additionally, the system demonstrates **cross-task transfer** to Text-to-SQL (Spider benchmark), where Qwen3.5-0.8B reaches 50% execution accuracy — outperforming a larger Qwen2.5-7B-Instruct zero-shot baseline (35%) without task-specific retraining.

---

## 6. Training

- **Base model**: Qwen3.5-0.8B
- **Method**: Flow-GRPO + LoRA (rank 8, <1% trainable parameters)
- **Reward**: hybrid (format + correctness) evaluated at trajectory level
- **Group size**: 8 trajectories per prompt, compared via group-relative advantage
- **Target**: improving multi-step decision quality, not just output text

Training ran on Modal using TRL's `GRPOTrainer` with PEFT; the merged adapter is served for evaluation on a single H200 GPU via a local FastAPI endpoint.

---

## 7. Quick Start

```bash
# install dependencies
pip install -r requirements_lora_serve.txt

# quick inference sanity check
python quick_start.py

# serve the LoRA-trained planner on a GPU node
python serve_lora_local.py

# run a benchmark (example: Bamboogle)
cd test/bamboogle && bash run.sh
```

For cluster deployment (SLURM + H200), see `run_lora_h200.sbatch` and `cluster_check.sh`.

---

## 8. Extending

The framework can be extended by defining a new tool interface and updating the planner's action space. Each role (Planner / Executor / Verifier) can be swapped or re-optimized independently because the interfaces between them are structured.

This makes the architecture applicable beyond QA and SQL tasks to domains such as robotics or workflow automation, where an agent must plan actions, interact with external environments, and verify outcomes in a feedback loop.

---

## 9. Credits

This project builds on the AgentFlow framework introduced in:

> AgentFlow: In-the-Flow Agentic System Optimization (ICLR 2026). [arXiv:2510.05592](https://arxiv.org/abs/2510.05592)

Contributions in this repository include: the LoRA + Flow-GRPO training pipeline on Qwen3.5-0.8B (TRL-based), the H200 cluster serving stack (FastAPI + SLURM), and the Text-to-SQL capability extension.
