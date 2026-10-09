# 远程部署、代码库地图与后续实现清单

本文面向当前 `gsm8k-agentic-rl` 工作区，而不是仅面向 GitHub 已提交版本。当前工作区约 2 GB，包含未提交的 GSM8K 代码、数据、MiniMind 模型、checkpoint 和实验输出。

## 1. 当前远程机器结论

- 主机：远程 8×RTX 4090 D 服务器，地址不写入公开仓库；每卡约 24 GiB。
- 驱动：550.144.03；适合 CUDA 12.4 wheel。
- GPU 0--3 属于 NUMA 0，GPU 4--7 属于 NUMA 1；组内为 PIX，跨组为 SYS；没有 NVLink。
- 系统盘约 2 TB，当前剩余约 692 GB；内存约 708 GiB。
- 检查时 8 张卡都在运行 `legged_lab` 任务。部署代码不等于可以立即启动训练；必须先获得空闲卡。

原则：MiniMind-64M 用一卡一实验；不要把 8 张卡用于一个 64M DDP 任务。8 卡优先并行算法/种子。Qwen3-4B 的 SFT 可先单卡 LoRA/QLoRA，RL 再根据策略、reference 与 rollout 显存实测采用 1--2 卡一组。

## 2. 从本地迁移当前工作区

只 `git clone` 会遗漏当前 16 项修改/未跟踪内容，因此首次迁移使用 `rsync`。

```bash
ssh -i ~/.ssh/mini-rl.pem root@<REMOTE_HOST> \
  'mkdir -p /root/projects/Mini-RL /root/mini-rl-data/{models,runs,cache}'

rsync -aH --info=progress2 \
  -e "ssh -i ~/.ssh/mini-rl.pem" \
  --exclude '.venv/' \
  --exclude '**/__pycache__/' \
  --exclude '.pytest_cache/' \
  --exclude 'models/minimind-3/.cache/' \
  /home/user/Documents/ChatGPT/Mini-RL/ \
  root@<REMOTE_HOST>:/root/projects/Mini-RL/
```

原理：`-aH` 保留时间、权限、软链接与硬链接；排除可重建缓存；不使用 `--delete`，避免后续同步时删除远端实验。checkpoint 和 `out/` 首次一并复制，是为了保留本地真实实验状态。以后最好将代码和运行产物分离。

迁移后核对：

```bash
ssh -i ~/.ssh/mini-rl.pem root@<REMOTE_HOST>
cd /root/projects/Mini-RL
git status --short
du -sh . checkpoints out models/minimind-3
sha256sum checkpoints/agent_sft_768.pth out/full_sft_768.pth
```

## 3. 创建隔离运行环境

不要使用服务器现有的机器人训练环境。

```bash
/root/miniconda3/bin/conda create -n mini-rl python=3.10 -y
/root/miniconda3/envs/mini-rl/bin/python -m pip install --upgrade pip setuptools wheel
/root/miniconda3/envs/mini-rl/bin/python -m pip install \
  torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
  --index-url https://download.pytorch.org/whl/cu124
cd /root/projects/Mini-RL
/root/miniconda3/envs/mini-rl/bin/python -m pip install -r requirements.txt
```

原理：驱动 550.144.03 高于 CUDA 12.4 GA 所需的 550.54.14；PyTorch 2.6 官方提供 cu124 wheel。wheel 自带 CUDA runtime，通常不需要另装完整 CUDA Toolkit。`requirements.txt` 故意没有启用 `torch`，所以必须先单独安装。

环境变量与缓存：

```bash
mkdir -p /root/mini-rl-data/cache/{huggingface,torch,wandb}
export HF_HOME=/root/mini-rl-data/cache/huggingface
export TORCH_HOME=/root/mini-rl-data/cache/torch
export WANDB_DIR=/root/mini-rl-data/cache/wandb
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8
```

建议把这些行写入项目专属的 `scripts/env/remote_4090.sh`，不要污染全局 shell。

## 4. 安装后验证

```bash
cd /root/projects/Mini-RL
PY=/root/miniconda3/envs/mini-rl/bin/python

$PY - <<'PY'
import torch
print('torch:', torch.__version__)
print('runtime CUDA:', torch.version.cuda)
print('GPU count:', torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(i, torch.cuda.get_device_name(i), torch.cuda.get_device_properties(i).total_memory)
print('bf16:', torch.cuda.is_bf16_supported())
PY

PYTHONPATH=. $PY -m unittest discover -s tests -p 'test_*.py'
PYTHONPATH=. $PY -m compileall -q model dataset trainer scripts tests
nvidia-smi topo -m
```

先测试 CPU/纯函数测试，再做单卡 smoke test。任何正式训练前都执行 `nvidia-smi`，不能因某卡尚有余量就与现有任务混跑。

## 5. 数据和单卡 smoke test

数据已随工作区迁移，但仍应重跑校验：

```bash
cd /root/projects/Mini-RL
PY=/root/miniconda3/envs/mini-rl/bin/python
$PY scripts/prepare/prepare_gsm8k.py
$PY scripts/prepare/prepare_gsm8k_agent_data.py
$PY scripts/evaluate/evaluate_gsm8k.py \
  --references data/processed/gsm8k/validation.jsonl \
  --predictions path/to/predictions.jsonl
```

训练脚本通常假定从仓库根目录启动，但 Python trainer 又会 `cd trainer`；路径语义必须保持不变。先选一张真正空闲的卡：

```bash
cd /root/projects/Mini-RL
export CUDA_VISIBLE_DEVICES=0
export MM_REPO_ROOT=/root/projects/Mini-RL
export MM_PYTHON_BIN=/root/miniconda3/envs/mini-rl/bin/python
export MM_SAVE_DIR=/root/mini-rl-data/runs/smoke
MM_MAX_UPDATES=3 MM_SEED=9001 bash scripts/run_math_dapo_optimization.sh
```

`CUDA_VISIBLE_DEVICES=0` 后，进程内部看到的该卡编号仍是 `cuda:0`。smoke test 要检查日志、JSONL 指标、checkpoint、断点恢复以及训练结束后显存释放。

## 6. 8 卡并发方式

MiniMind 正式矩阵应是一卡一进程，并给每个 run 独立目录。例如用 tmux：

```bash
tmux new-session -d -s mm_grpo_s42 \
  'cd /root/projects/Mini-RL && CUDA_VISIBLE_DEVICES=0 MM_SEED=42 MM_MAX_UPDATES=30 bash scripts/run_one_rlvr.sh grpo'
tmux new-session -d -s mm_cispo_s42 \
  'cd /root/projects/Mini-RL && CUDA_VISIBLE_DEVICES=1 MM_SEED=42 MM_MAX_UPDATES=30 bash scripts/run_one_rlvr.sh cispo'
tmux new-session -d -s mm_dapo_s42 \
  'cd /root/projects/Mini-RL && CUDA_VISIBLE_DEVICES=2 MM_SEED=42 MM_MAX_UPDATES=30 bash scripts/run_one_rlvr.sh dapo'
tmux new-session -d -s mm_gspo_s42 \
  'cd /root/projects/Mini-RL && CUDA_VISIBLE_DEVICES=3 MM_SEED=42 MM_MAX_UPDATES=30 bash scripts/run_one_rlvr.sh gspo'
```

这里的 `run_one_rlvr.sh` 是后续必须新增的脚本；现有 `run_rlvr_ablation.sh` 会串行遍历全部算法和 seed，不适合把矩阵拆给 8 张卡。不要同时启动多个现有 ablation 脚本，因为它们会争用相同 `out/` 文件名。

如果确实运行 DDP，组内优先 `0,1,2,3` 或 `4,5,6,7`，避免跨 NUMA：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 trainer/train_full_sft.py ...
```

DDP 每个 rank 都复制完整模型，只做数据并行，不解决单卡放不下的问题；模型放不下时应使用 FSDP/ZeRO，而不是增加 DDP 卡数。

## 7. 代码来源分层

权威来源记录在 `docs/foundation_model_interview/SOURCES.md`：工程基线是 `jingyaogong/minimind` 提交 `393e387e...`，Apache-2.0。模型、基础数据集、Pretrain/SFT/LoRA/DPO/PPO/GRPO 框架主要来自上游；统一策略目标、DAPO/GSPO、RLVR verifier、审计、评测、实验协议与 GSM8K 新主线是本项目扩展。论文只提供算法定义，项目代码是在 MiniMind 张量与 rollout 接口上重写，并非复制 verl/DAPO/VerlTool 源码。

## 8. 仓库文件逐项地图

### 根目录

- `.gitignore`：忽略数据、权重、日志、缓存；当前分支扩充。
- `LICENSE`、`CODE_OF_CONDUCT.md`：MiniMind 上游 Apache-2.0 与社区规范。
- `README.md`：当前中文项目入口，已改写为 GSM8K Agentic RL 主线。
- `README_en.md`：MiniMind 上游英文说明，尚未与新中文 README 完全同步。
- `requirements.txt`：Python 依赖；由上游依赖演化而来，torch 被刻意注释，且 `jsonlines` 重复，后续应锁版本。
- `eval_llm.py`：上游通用 Transformers 模型交互评测入口。
- `patches/cached_sdpa_fastpath.patch`：本项目的 KV cache/SDPA 性能实验补丁，不是当前自动应用步骤。

### `model/`

- `__init__.py`：Python 包标记。
- `model_minimind.py`：上游 MiniMind Decoder、RMSNorm、RoPE、GQA、SDPA、SwiGLU/MoE 与生成逻辑；项目做过架构验证相关调整。
- `model_lora.py`：上游自定义 LoRA 注入、保存和加载，不是 Hugging Face PEFT。
- `tokenizer.json`、`tokenizer_config.json`：MiniMind tokenizer 资产，来自上游模型包。

### `dataset/`

- `__init__.py`：包标记。
- `dataset.md`：上游数据放置说明。
- `lm_dataset.py`：上游 Pretrain/SFT/DPO/RL 数据集类，当前工作区增加 GSM8K Agent SFT 兼容。
- `gsm8k.py`：当前未提交的新主线模块；规范化 GSM8K、固定切分、泄漏检查、JSONL 和 SHA-256 manifest。
- `manifests/README.md`：manifest 规则说明。
- `manifests/gsm8k.json`：当前本地 GSM8K 原始/规范化切分的数量与哈希，生成产物。
- `manifests/gsm8k_agent.json`：Agent SFT/RL 切分的数量与哈希，生成产物。

### `trainer/`

- `trainer_utils.py`：上游训练公共底座；DDP 初始化、seed、学习率、checkpoint、模型/tokenizer 加载，项目加强了原子保存和恢复。
- `train_tokenizer.py`：上游 tokenizer 训练入口。
- `train_pretrain.py`：上游从零预训练入口。
- `train_full_sft.py`：上游全参 SFT；当前工作区增加 Agent assistant-only mask/数据兼容。
- `train_lora.py`：上游 MiniMind 自定义 LoRA 训练。
- `train_dpo.py`：上游 DPO 主体，项目修复累积窗口/保存并增加配套评测。
- `train_ppo.py`：上游 PPO 框架，项目增加 rollout 与训练正确性修正。
- `train_grpo.py`：上游 GRPO 文件上扩展；旧入口，核心新算法逐步集中到统一模块。
- `train_distillation.py`：上游蒸馏入口，项目补充 FP32 KL、对照和累积修正。
- `train_agent.py`：上游 Agent RL 文件上的核心扩展；多轮工具轨迹、PPO/GRPO/CISPO/DAPO/GSPO、动态采样、奖励、DDP、日志与 checkpoint。当前仍与 MiniMind 配置强绑定。
- `rollout_engine.py`：本项目抽象的 rollout engine；封装 PyTorch 生成、行为策略 log-prob 与轨迹结构。
- `policy_optimization.py`：本项目原创教学实现；统一 GRPO、CISPO、DAPO、GSPO 目标和诊断量。
- `experiment_logging.py`：本项目 JSONL 指标与 run config 记录。
- `math_env.py`：当前未提交的新模块；AST 白名单 calculator、GSM8K answer parser 和数值 verifier，供训练与评测共用。

### `scripts/`：数据、审计和评测 Python 程序

- `download/download_gsm8k.py`：当前未提交；下载 GSM8K 到 `data/raw`。
- `prepare/prepare_gsm8k.py`：当前未提交；生成固定 train/validation/test 与 manifest。
- `prepare/prepare_gsm8k_agent_data.py`：当前未提交；生成 Agent SFT/RL 数据。
- `evaluate/evaluate_gsm8k.py`：当前未提交；按 ID 对预测做固定集数值评测。
- `prepare_agent_sft_data.py`：从 verified tool/math 数据构建冷启动 SFT。
- `prepare_dpo_data.py`：构建按 prompt 隔离的 DPO train/holdout。
- `prepare_kd_data.py`：构建知识蒸馏数据。
- `prepare_tool_rlvr_data.py`：生成确定性 mock tool RLVR 数据。
- `prepare_tool_rlvr_challenge_data.py`：生成非饱和组合工具挑战集。
- `split_agent_dataset.py`：按 prompt/问题分组拆分 Agent train/eval。
- `audit_sft_mask.py`：检查只在 assistant token 上计算 SFT loss。
- `audit_lora_split.py`：检查 LoRA train/holdout 隔离。
- `audit_moe_routing.py`：检查 MoE 路由和负载均衡。
- `audit_multiturn_serialization.py`：检查多轮 assistant/tool/observation 序列化与 mask。
- `audit_tool_rlvr_dataset.py`：全量重放工具参数、结果与 train/eval 泄漏。
- `benchmark_architecture.py`：比较 cache/非 cache 解码速度与峰值显存。
- `verify_flash_sdpa.py`：验证 PyTorch SDPA 后端选择条件。
- `verify_model_flash_sdpa.py`：在 MiniMind 真实模型上验证 SDPA/显存。
- `inspect_rl_learnability.py`：在训练前测奖励方差、有效组率与可学习性。
- `eval_agent_rlvr.py`：统一固定集 Agent RLVR 逐轨迹评测。
- `eval_distillation.py`：蒸馏模型独立 holdout 评测。
- `eval_dpo_holdout.py`：DPO preference margin holdout 评测。
- `eval_lora_holdout.py`：LoRA holdout 评测。
- `eval_toolcall.py`：上游/扩展的工具调用交互评测。
- `summarize_rl_metrics.py`：聚合训练 JSONL。
- `summarize_eval_results.py`：聚合固定集结果与相对 baseline 增益。
- `plot_dapo_learnability_curve.py`：绘制 DAPO 可学习性曲线。
- `collect_experiment_evidence.py`：采集 commit、环境、日志、hash 与退出状态。
- `convert_model.py`：上游 PyTorch/Hugging Face 格式转换。
- `chat_api.py`：调用 OpenAI-compatible API 的聊天客户端。
- `serve_openai_api.py`：将本地 MiniMind 暴露为 OpenAI 风格 API。
- `web_demo.py`：Streamlit Web 演示。

`.gitkeep` 文件只用于让空的 `download/prepare/train/evaluate` 目录进入 Git，没有运行逻辑。

### `scripts/`：Shell 流水线

- `run_student_pretrain.sh`：512 hidden student 预训练。
- `run_student_sft.sh`：student 全参 SFT。
- `run_agent_sft.sh`：共同 Agent 冷启动 SFT。
- `run_kd_ablation.sh`：蒸馏与 CE-only 对照。
- `run_kd_pipeline_after_pretrain.sh`：等待预训练后执行 KD 链。
- `run_dpo_after_readiness.sh`：readiness 通过后执行 DPO 与 holdout。
- `run_rlvr_readiness_after_kd.sh`：等待 KD 后准备数据、Agent SFT 并评估有效组率。
- `run_rlvr_pilot_after_dpo.sh`：等待 DPO 后运行小预算 RLVR pilot。
- `run_architecture_verification_after_pilot.sh`：pilot 后跑 cache/SDPA/MoE 架构验证。
- `run_rlvr_formal_after_architecture.sh`：架构验证后串行运行正式三 seed 消融。
- `run_finalize_after_formal.sh`：汇总、测试、hash 和最终证据采集。
- `run_rlvr_ablation.sh`：串行遍历 seed × GRPO/CISPO/DAPO/GSPO。
- `run_rlvr_eval.sh`：统一 holdout 和 decode seeds 评测 RL checkpoint。
- `run_dapo_learnability_diagnostic.sh`：DAPO 有效组诊断。
- `run_math_dapo_optimization.sh`：修正后的数学 DAPO 单实验。
- `run_math_optimization_eval.sh`：数学优化 checkpoint 对照评测。
- `run_tool_corrected_ablation.sh`：修正后的工具 GRPO/CISPO/GSPO 分支。
- `run_tool_corrected_ablation_eval.sh`：工具消融三 decode seed 评测。

这些 shell 中一部分是为旧单卡 `/root/minimind` 环境写的，包含 `screen` 等待、成功 marker 和硬编码 Python 路径。它们记录了真实实验编排，但不应直接作为新 8 卡调度器。

### `tests/`

- `test_dpo_objective.py`：DPO loss/更新行为。
- `test_distillation.py`：KL、CE 与累积边界。
- `test_architecture_features.py`：GQA/MoE/架构特性。
- `test_kv_cache_consistency.py`：cache 与非 cache logits/生成一致性。
- `test_agent_rewards.py`：格式、工具执行、答案和严格奖励。
- `test_agent_sft_data.py`：Agent SFT 序列与 assistant-only mask；当前有本地修改。
- `test_rollout_probabilities.py`：rollout 行为策略 log-prob 一致性。
- `test_policy_optimization.py`：GRPO/CISPO/DAPO/GSPO 公式与边界。
- `test_gsm8k_pipeline.py`：当前未提交；GSM8K 切分、泄漏、calculator 和 verifier。
- `fixtures/agent_rl_smoke.jsonl`：小型确定性测试夹具。

### `docs/`

- `PROJECT_PLAN.md`：两阶段研究计划。
- `DATASET_AND_BENCHMARK.md`：GSM8K 数据、切分、污染和 benchmark 规范。
- `EXPERIMENT_PROTOCOL.md`：共同初始化、公平对照、seed 与报告规则。
- `WORKLOG.md`：按日期记录已经做过和未做的工作。
- `REMOTE_DEPLOYMENT_AND_REPOSITORY_MAP.md`：本文。
- `foundation_model_interview/README.md`：旧研究文档索引。
- `00_END_TO_END_GUIDE.md`：旧主线端到端执行指南。
- `01_TRAINING_PIPELINE_REPORT.md`：Pretrain/SFT/LoRA/KD/DPO 训练链报告。
- `02_AGENTIC_RLVR_FROM_ZERO_TO_MASTERY.md`：Agentic RLVR 教程。
- `02_POLICY_OPTIMIZATION_REPORT.md`：早期策略优化报告。
- `03_AGENTIC_RL_RLVR_REPORT.md`：早期 Agent RL 报告。
- `03_POLICY_OPTIMIZATION_FROM_ZERO_TO_MASTERY.md`：统一策略算法详解。
- `04_EXPERIMENT_PROTOCOL.md`：旧实验真实性协议。
- `05_ACTUAL_EXPERIMENT_RESULTS.md`：已有真实结果和负结果。
- `06_FINAL_PROJECT_REPORT.md`：旧主线总结。
- `07_RESUME_ALIGNED_TECHNICAL_REPORT.md`：面向简历的技术边界说明。
- `08_EXPERIMENTAL_NARRATIVE.md`：实验叙事。
- `09_RL_OPTIMIZATION_FOLLOWUP.md`：精度、log-prob 与 Agent-SFT 冷启动训练修正后的后续实验。
- `SOURCES.md`：上游、论文、官方实现与本项目原创边界的权威清单。

### Notebook、图片和模型资产

- `learning_notebooks/01`--`07` 与 `torch用法.ipynb`：MiniMind/本项目演化形成的 tokenizer、embedding、优化、MoE、反传、架构、预训练和微调教学。
- `08_Agentic_RLVR...ipynb`、`09_GRPO...ipynb`、`10_简历项目...ipynb`：本项目新增的 Agentic RLVR、策略算法和项目报告 notebook。
- `learning_notebooks/samples/*.jsonl`：notebook 最小示例数据。
- `images/*`：MiniMind 上游 README/报告图片；不是本项目的新实验结果证据。
- `models/minimind-3/*`：从 Hugging Face/模型仓库下载的完整 MiniMind-3 模型包；`model.safetensors` 是权重，其余 JSON/Jinja 是配置、tokenizer 和 chat template，README 与 `images/` 是上游说明资产。
- `models/minimind-3/.cache/huggingface/download/*.lock`、`*.metadata`：Hugging Face 下载缓存和元数据，不属于源码，不应迁移或提交。

### 数据、checkpoint 与输出

- `data/raw/gsm8k/official_{train,test}.jsonl`：外部 GSM8K 原始数据。
- `data/processed/gsm8k/{train,validation,test}.jsonl`：本项目固定切分生成物。
- `data/processed/gsm8k_agent/{train_sft,train_rl,validation_rl,test_rl}.jsonl`：Agent 模板化生成物。
- `checkpoints/*.pth`：完整恢复 checkpoint，含 optimizer/step 等；本地实验产物。
- `out/*.pth`、`out/rl_pilot/**/*.pth`：模型权重/阶段 checkpoint。
- `out/logs/*.log`：训练与评测 stdout/stderr。
- `out/metrics/*.jsonl`：逐 update 结构化指标。
- `out/eval_*/summary.csv`、`trajectories.jsonl`：固定集汇总与逐轨迹审计。
- `out/run_meta/*`：commit、dirty status、pip freeze、GPU 信息、数据/checkpoint hash。
- `configs/minimind_64m/.gitkeep`、`configs/qwen3_4b/.gitkeep`：目前只是空目录占位，说明配置文件尚未真正落地。
- `results/README.md`：规定只提交审核后的小结果表、manifest 和图，不提交原始大产物。

## 9. 之后必须新增或重构的文件

按优先级排序：

1. `environment/requirements-lock.txt` 或 `pyproject.toml` + lock：冻结经过验证的完整环境，去掉重复依赖。
2. `scripts/env/remote_4090.sh`：统一仓库、Python、缓存、数据和运行根目录。
3. `configs/minimind_64m/{sft,grpo,cispo,dapo,gspo,ppo}.yaml`：把散落在 shell 的超参数配置化。
4. `scripts/run_one_rlvr.sh`：只运行一个 algorithm × seed × GPU，使用唯一 run ID；这是 8 卡并发的关键。
5. `scripts/launch_experiment_matrix.py`：读取矩阵、检查空闲 GPU、启动/恢复任务、记录 PID 和状态；不要杀死非本项目进程。
6. `scripts/check_remote_readiness.py`：检查 GPU 占用、磁盘、数据/checkpoint hash、环境和写权限。
7. `trainer/model_adapter.py`：统一 MiniMind 与 Hugging Face CausalLM 的 forward/generate/log-prob/checkpoint 接口。
8. `trainer/qwen3_adapter.py`：Qwen3 tokenizer/chat template、thinking 模式、tool schema 和生成接口。
9. `trainer/train_qwen_lora_sft.py`：PEFT LoRA/QLoRA、gradient checkpointing、BF16 与 adapter 保存。
10. `trainer/train_qwen_rl.py`：将统一 RL 目标接到 Qwen actor/reference；明确 rollout 与训练权重同步。
11. `configs/qwen3_4b/{lora_sft,grpo,ppo}.yaml`：Qwen 显存和序列长度配置。
12. `tests/test_qwen_adapter.py`、`tests/test_qwen_tool_serialization.py`、`tests/test_distributed_smoke.py`：适配器、工具模板和多进程最小测试。
13. `scripts/evaluate/evaluate_model_gsm8k.py`：直接加载任一 adapter/checkpoint 生成并评测，取代手工拼 predictions。
14. `scripts/summarize_experiment_matrix.py`：按算法与训练 seed 聚合均值、标准差、GPU-hour 和 token budget。
15. `docs/REMOTE_RUNBOOK.md`：故障恢复、续训、产物归档、卡被抢占和 OOM 调参规则。

此外应重构 `run_*_after_*.sh`：去掉 `/root/minimind`、特定 Conda 路径、`screen -ls` 和轮询前置任务的耦合，改为显式配置与可重入状态文件。

## 10. 当前不能忽略的风险

- 工作区是 dirty 状态；正式远端实验前应提交当前 GSM8K 改动，或至少保存 `git diff` 和未跟踪文件 hash。
- 现有 8 卡均被其他训练占用，不能直接启动。
- Stage 2 Qwen3-4B 目前只有目录和计划，没有可运行实现。
- 现有多卡是 DDP，不能把大于单卡显存的模型自动切到多卡。
- 多个旧 shell 默认写同名 `out/`/`checkpoints/`，并发会冲突。
- 本地 Hugging Face `.cache`、原始日志和大 checkpoint 不应进入普通 Git。
- `README_en.md`、旧 foundation 文档与新 GSM8K 主线存在时间层次，引用结果时必须以产物、hash 和 `05_ACTUAL_EXPERIMENT_RESULTS.md` 为准。
