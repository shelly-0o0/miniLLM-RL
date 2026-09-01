# miniLLM-RL

A reproducible project for lightweight foundation-model training and Agentic RLVR.

This repository uses [MiniMind](https://github.com/jingyaogong/minimind) as the lightweight language-model baseline. Its focus is my engineering work on training-pipeline enhancements, policy optimization, tool use, verifiable rewards, fixed-set evaluation, and experiment reporting.

## Highlights

- Pretraining, assistant-only SFT, LoRA, knowledge distillation, and DPO
- Multi-turn Tool-Use Agent environments and Agentic RLVR
- Unified GRPO, CISPO, DAPO, and GSPO policy optimization
- Leakage-aware train/eval splits and fixed-set multi-seed evaluation
- Trajectory serialization, action-mask, verifier, and behavior-policy audits
- Architecture verification for KV Cache, GQA, MoE routing, and Flash SDPA
- Reproducible local JSONL metrics and experiment evidence

## Repository layout

```text
model/                         Model structure, LoRA, and tokenizer
trainer/                       Pretraining, SFT, DPO, RL, and distillation
scripts/                       Data, training, evaluation, and audit scripts
tests/                         Regression tests
docs/foundation_model_interview/  Technical reports and experiment protocol
learning_notebooks/            Executable learning and experiment notebooks
patches/                       Performance patches
```

## Documentation

- [Documentation index](docs/foundation_model_interview/README.md)
- [End-to-end execution guide](docs/foundation_model_interview/00_END_TO_END_GUIDE.md)
- [Agentic RLVR tutorial](docs/foundation_model_interview/02_AGENTIC_RLVR_FROM_ZERO_TO_MASTERY.md)
- [Policy optimization tutorial](docs/foundation_model_interview/03_POLICY_OPTIMIZATION_FROM_ZERO_TO_MASTERY.md)
- [Actual experiment results](docs/foundation_model_interview/05_ACTUAL_EXPERIMENT_RESULTS.md)
- [Final project report](docs/foundation_model_interview/06_FINAL_PROJECT_REPORT.md)

## Quick start

```bash
git clone https://github.com/shelly-0o0/miniLLM-RL.git
cd miniLLM-RL
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
PYTHONPATH=. python -m unittest discover -s tests -p 'test_*.py'
```

Install PyTorch separately according to your operating system and CUDA environment. Full datasets and checkpoints are intentionally excluded from Git; see [dataset/dataset.md](dataset/dataset.md) and the execution guide.

## Attribution

This project is based on [jingyaogong/minimind](https://github.com/jingyaogong/minimind). See [LICENSE](LICENSE) and [SOURCES.md](docs/foundation_model_interview/SOURCES.md) for licensing and source mapping.
