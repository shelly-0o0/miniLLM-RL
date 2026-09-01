# 从服务器到可量化结果：端到端执行指南

本文所有训练命令都以官方仓库的目录约定为准：训练脚本在 `trainer/` 内执行，权重写到 `out/`，完整恢复状态写到 `checkpoints/`。新增评测脚本从仓库根目录执行。

## 0. 复现目标与推荐资源

最低可用配置：Ubuntu 22.04、NVIDIA 24GB GPU、64GB 内存、200GB 数据盘。该配置适合 64M dense 模型的 mini 数据、单卡算法验证。

面试证据配置：4×24GB 或 4×40/80GB GPU、128GB 内存、500GB NVMe。多卡用于 Pretrain/SFT DDP；RL 首轮建议先用单卡 Torch rollout 验证公式与奖励，再扩展到多卡或 SGLang。

需要记录的服务器信息：

```bash
nvidia-smi
nvcc --version
lsb_release -a
df -h
free -h
```

这些输出用于解释驱动、CUDA、显存和磁盘约束。不要把云厂商镜像中的 CUDA Toolkit 版本与 PyTorch wheel 自带的 CUDA runtime 混为一谈；真正先看 `torch.version.cuda` 和 `torch.cuda.is_available()`。

## 1. 创建服务器与基础环境

在云控制台创建 Ubuntu 22.04 GPU 实例，开放 SSH 端口即可。训练和 SGLang 服务不应直接暴露到公网；如需远程查看，用 SSH 端口转发。

服务器内执行：

```bash
sudo apt-get update
sudo apt-get install -y git git-lfs build-essential screen tmux htop nvtop
git lfs install
```

安装 Miniconda 后创建隔离环境。若镜像已有 conda，可直接从 `conda create` 开始：

```bash
conda create -n minimind python=3.10 -y
conda activate minimind
python -m pip install --upgrade pip setuptools wheel
```

以下组合与仓库注释中的 PyTorch 2.6 对齐；若服务器驱动不支持 CUDA 12.4，应从 PyTorch 官方安装选择器换成匹配版本：

```bash
python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
```

验证：

```bash
python - <<'PY'
import torch
print("torch", torch.__version__)
print("torch CUDA runtime", torch.version.cuda)
print("CUDA available", torch.cuda.is_available())
print("GPU count", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(i, torch.cuda.get_device_name(i), torch.cuda.get_device_properties(i).total_memory / 2**30, "GiB")
PY
```

解释：PyTorch 能列出 GPU 才进入训练；`nvcc` 不存在不等于 PyTorch 不能使用 GPU，因为 wheel 带运行时，但编译某些 fused kernel 时仍需要完整 Toolkit。

## 2. 获取严格的官方基线和本次增强代码

官方仓库：

```bash
git clone https://github.com/jingyaogong/minimind.git
cd minimind
git rev-parse HEAD
git remote -v
```

本次开发核对的基线为 `393e387e9ad99f0f04c296e4c5e7353f4444629f`。核对时官方较新的 `d65ef2c00ebc6082f9df11541e1b191655eddb00` 仅改变 README，训练和模型代码相同。为了让代码行级一致，可固定前者：

```bash
git checkout 393e387e9ad99f0f04c296e4c5e7353f4444629f
```

本地工作区已经包含本次增强。将它同步到服务器最直接：

```bash
rsync -av --exclude '.git' /本机/minimind/ 用户名@服务器IP:/训练目录/minimind/
```

同步后在服务器检查新增文件：

```bash
cd /训练目录/minimind
test -f trainer/policy_optimization.py
test -f scripts/run_rlvr_ablation.sh
test -f scripts/eval_agent_rlvr.py
```

建议在自己的分支提交，保证实验和代码可追溯：

```bash
git switch -c interview-rlvr
git add trainer/policy_optimization.py trainer/experiment_logging.py trainer/train_grpo.py trainer/train_agent.py scripts tests docs
git commit -m "add auditable DAPO GSPO and agent RLVR experiments"
git rev-parse HEAD
```

此提交是个人增强版本，不应说成 MiniMind 上游官方提交。

## 3. 安装项目依赖并运行预检

```bash
python -m pip install -r requirements.txt
python -m pip install "huggingface_hub[cli]"
python -m unittest discover -s tests -v
python -m compileall model dataset trainer scripts
```

当前增强版最终单测共 31 个，覆盖模型架构与 KV Cache、KD/DPO 目标、四种策略 loss 的有限性/梯度、动态组筛选、软超长惩罚、工具 verifier、多轮 action mask、数值子串 reward-hacking 防护，以及禁止 rollout 后左截断的回归保护。

运行前保存环境快照：

```bash
mkdir -p out/run_meta
python -m pip freeze > out/run_meta/pip_freeze.txt
nvidia-smi -q > out/run_meta/nvidia_smi.txt
git rev-parse HEAD > out/run_meta/git_commit.txt
```

### 3.1 所有长任务统一放入 `screen`

SSH 前台断开会向子进程传播 hangup。Pretrain、SFT、KD、DPO、RLVR 和长评测都必须在 `screen` 中运行，并把 stdout/stderr 写入日志。交互式做法是：

```bash
screen -S minimind_pretrain
cd /root/minimind/trainer
set -o pipefail
export PYTHONUNBUFFERED=1
# 在这里执行本指南的 Pretrain 命令，并在末尾加：
# 2>&1 | tee ../out/logs/pretrain_dense_mini.log
```

按 `Ctrl-A`、再按 `D` 脱离；之后可安全退出 SSH。重新连接后：

```bash
screen -ls
screen -r minimind_pretrain
tail -f /root/minimind/out/logs/pretrain_dense_mini.log
ps -eo pid,ppid,stat,etime,cmd | grep -E 'torchrun|train_|eval_' | grep -v grep
```

也可一次性后台启动脚本：

```bash
screen -dmS 任务名 bash -lc \
  'cd /root/minimind && set -o pipefail && bash scripts/某流水线.sh 2>&1 | tee out/logs/某流水线.log'
```

不要依赖镜像一定存在 `/usr/bin/time`；需要计时时使用 Bash 的 `SECONDS`、日志时间戳或先安装 GNU time。`screen -ls` 只能证明会话存在，是否还在计算要同时看 `ps`、日志增长和 `nvidia-smi`。下文为清晰起见展示训练命令本体，所有预计超过数分钟的命令均应在上述 `screen` 会话内执行。

## 4. 下载官方数据

快速复现至少需要：

```bash
hf download jingyaogong/minimind_dataset \
  pretrain_t2t_mini.jsonl \
  sft_t2t_mini.jsonl \
  rlaif.jsonl \
  agent_rl.jsonl \
  agent_rl_math.jsonl \
  dpo.jsonl \
  --repo-type dataset \
  --local-dir dataset
```

国内网络也可用仓库要求的 ModelScope：

```bash
modelscope download --dataset gongjy/minimind_dataset --local_dir ./dataset
```

ModelScope 命令会下载整个数据仓库；Hugging Face 命令只取列出的文件。完整主线把 mini 文件换成 `pretrain_t2t.jsonl` 和 `sft_t2t.jsonl`。

检查文件和 JSONL：

```bash
ls -lh dataset/*.jsonl
python - <<'PY'
import json
from pathlib import Path
for name in ["pretrain_t2t_mini.jsonl", "sft_t2t_mini.jsonl", "agent_rl_math.jsonl"]:
    path = Path("dataset") / name
    with path.open(encoding="utf-8") as f:
        first = json.loads(next(f))
    print(name, first.keys())
PY
```

按题目哈希持久化划分，保证完全相同的 user 问题不会跨训练/评测集：

```bash
python scripts/split_agent_dataset.py \
  --input dataset/agent_rl_math.jsonl \
  --train_output dataset/agent_rl_math_train.jsonl \
  --eval_output dataset/agent_rl_math_eval.jsonl \
  --manifest out/run_meta/agent_split_manifest.json \
  --eval_ratio 0.1 \
  --seed 42
```

正式实验固定并归档 manifest。该脚本只能防精确题目重复；语义改写、同模板换数字和预训练污染仍需 MinHash/embedding 近重复检查及人工抽样。

## 5. 先确认架构，不重复造上游已有模块

官方 `model/model_minimind.py` 已实现：

- Decoder-only causal LM：`MiniMindModel`、`MiniMindBlock`、`MiniMindForCausalLM`。
- RMSNorm：主干 Pre-Norm、输出 Norm 和 Q/K Norm。
- RoPE 与 YaRN：`precompute_freqs_cis`；`inference_rope_scaling=True` 时使用 YaRN。
- GQA：默认 8 个 Query head、4 个 KV head，`repeat_kv` 只做逻辑扩展。
- KV Cache：每层保存 `(K,V)`，生成时仅输入新增 token。
- SwiGLU：`SiLU(gate_proj(x)) * up_proj(x)` 再降维。
- Flash Attention 路径：调用 PyTorch `scaled_dot_product_attention`；这不是直接依赖 `flash-attn` Python 包。
- 稀疏 MoE：4 个专家、Top-1 路由和负载均衡辅助损失。

现有 MoE 是教学级单进程实现，没有 expert parallel、capacity factor、token dispatch all-to-all 或 routing replay。面试时必须说明边界。

量化 KV Cache 和推理吞吐：

```bash
python scripts/benchmark_architecture.py \
  --weight full_sft \
  --prompt_length 1024 \
  --decode_tokens 128 \
  --flash_attn 1 \
  --output out/architecture_benchmark.csv
```

做消融：

```bash
python scripts/benchmark_architecture.py --weight full_sft --prompt_length 1024 --decode_tokens 128 --flash_attn 0 --output out/architecture_benchmark.csv
python scripts/benchmark_architecture.py --weight none --prompt_length 1024 --decode_tokens 128 --num_key_value_heads 8 --output out/architecture_benchmark.csv
python scripts/benchmark_architecture.py --weight full_sft --prompt_length 4096 --decode_tokens 128 --inference_rope_scaling 1 --output out/architecture_benchmark.csv
```

第二条改变 KV head 数会改变参数形状，不能加载原 4-KV-head checkpoint；只应用于 `--weight none` 的结构基准，或重新预训练 MHA 变体。对已训练 GQA 模型，使用脚本输出的理论 MHA cache 做公平公式对照。

GQA 理论 KV Cache 字节数为：

```text
2 × layers × sequence_length × kv_heads × head_dim × bytes_per_element
```

默认 4 KV heads 相对 8-head MHA 的 KV Cache 理论降幅为 50%。这只是结构确定值；实际峰值还包含权重、激活、allocator 和 attention workspace，应报告脚本实测。

YaRN 的 4096-token 命令只验证长上下文执行路径和资源，不证明长文本质量。正式结论还需做不同长度的 validation PPL、passkey/needle retrieval 和长文任务，并与未缩放 RoPE 对照；如质量下降，应在长序列数据上做小学习率 continued pretraining，而不是只打开推理 flag。

## 6. Pretrain：从随机初始化开始

Mini 数据、单卡：

```bash
cd trainer
torchrun --standalone --nproc_per_node 1 train_pretrain.py \
  --data_path ../dataset/pretrain_t2t_mini.jsonl \
  --save_weight pretrain \
  --hidden_size 768 \
  --num_hidden_layers 8 \
  --max_seq_len 768 \
  --batch_size 8 \
  --accumulation_steps 4 \
  --dtype bfloat16 \
  --epochs 2 \
  --from_weight none \
  --save_interval 500
```

解释：每卡 micro batch 为 8；单卡累积 4 次后有效 batch 为 32 条。多卡时全局有效 batch 近似为：

```text
per_gpu_batch × accumulation_steps × world_size
```

核心命令默认不启用外部实验平台；若已经配置账号与 API key，再自行追加 `--use_wandb`。无论是否使用平台，都应保留本地 `tee` 日志、JSONL 指标和 checkpoint。

多卡：

```bash
torchrun --standalone --nproc_per_node 4 train_pretrain.py \
  --data_path ../dataset/pretrain_t2t_mini.jsonl \
  --batch_size 8 \
  --accumulation_steps 1 \
  --dtype bfloat16 \
  --max_seq_len 768
```

四卡的全局 batch 同样为 32，便于与单卡累积对照。DDP 每卡持有完整模型副本，`DistributedSampler` 切数据，反向时 all-reduce 梯度；它不是模型分片。

输出：`out/pretrain_768.pth`；恢复：

```bash
torchrun --standalone --nproc_per_node 4 train_pretrain.py --from_resume 1
```

恢复时保持模型形状和数据路径一致。改变卡数虽有上游 step 换算，仍应重新检查有效 batch、学习率与数据顺序。

### 6.1 稀疏 MoE 分支

MoE 参数形状与 dense 不同，需要单独从头预训练并 SFT：

```bash
torchrun --standalone --nproc_per_node 4 train_pretrain.py \
  --data_path ../dataset/pretrain_t2t_mini.jsonl \
  --save_weight pretrain \
  --from_weight none \
  --hidden_size 768 \
  --num_hidden_layers 8 \
  --use_moe 1 \
  --batch_size 8 \
  --accumulation_steps 1 \
  --dtype bfloat16

torchrun --standalone --nproc_per_node 4 train_full_sft.py \
  --data_path ../dataset/sft_t2t_mini.jsonl \
  --from_weight pretrain \
  --save_weight full_sft \
  --hidden_size 768 \
  --num_hidden_layers 8 \
  --use_moe 1 \
  --batch_size 4 \
  --accumulation_steps 2 \
  --dtype bfloat16
```

输出分别为 `out/pretrain_768_moe.pth` 与 `out/full_sft_768_moe.pth`。上游是 Top-1、4 experts、aux load-balancing 的教学实现；DDP 会在每卡复制全部专家，没有 expert parallel 或 all-to-all。正式 MoE 报告至少补每专家 token 比例、router entropy、负载 CV、dropped token、总参数/激活参数与吞吐。

## 7. SFT：只训练 Assistant token

```bash
torchrun --standalone --nproc_per_node 4 train_full_sft.py \
  --data_path ../dataset/sft_t2t_mini.jsonl \
  --from_weight pretrain \
  --save_weight full_sft \
  --hidden_size 768 \
  --num_hidden_layers 8 \
  --max_seq_len 768 \
  --batch_size 4 \
  --accumulation_steps 4 \
  --learning_rate 1e-5 \
  --dtype bfloat16 \
  --epochs 2
```

`SFTDataset.generate_labels` 先将全部 label 设为 `-100`，只恢复 assistant 区间；`MiniMindForCausalLM` 做 next-token shift，`cross_entropy(ignore_index=-100)` 忽略 system/user/tool/pad。必须抽样解码 label 做人工核对：若模板的 assistant 起止 token 与 tokenizer 不一致，loss 可能看似正常但训练目标已错。

自动审计前 64 条样本的 mask、pad、监督 token 和可读 assistant span：

```bash
cd ..
python scripts/audit_sft_mask.py \
  --data_path dataset/sft_t2t_mini.jsonl \
  --samples 64 \
  --output out/run_meta/sft_mask_audit.json
```

任何 mismatch 或零监督样本都会非零退出。随后做简单对话检查：

```bash
python eval_llm.py --weight full_sft
```

## 8. LoRA

准备与 SFT 相同的 `conversations` JSONL，例如 `dataset/lora_domain.jsonl`，然后：

```bash
cd trainer
torchrun --standalone --nproc_per_node 1 train_lora.py \
  --data_path ../dataset/lora_domain.jsonl \
  --from_weight full_sft \
  --lora_name lora_domain \
  --hidden_size 768 \
  --num_hidden_layers 8 \
  --max_seq_len 768 \
  --batch_size 8 \
  --accumulation_steps 4 \
  --learning_rate 1e-4 \
  --dtype bfloat16 \
  --epochs 5
```

只保存 LoRA 参数为 `out/lora_domain_768.pth`。推理：

```bash
cd ..
python eval_llm.py --weight full_sft --lora_weight lora_domain
```

LoRA 脚本会冻结非 LoRA 参数；`torch.compile` 因 monkey patch 自动关闭。报告 trainable parameter 比例、峰值显存、吞吐和领域/通用双评测，不要只报训练 loss。

本项目的医疗 LoRA 还使用固定 90/10 划分。训练完成后，用同一次物化的 token 同时评估 base 和 adapter，避免 `post_processing_chat` 的随机性让两模型看到不同输入：

```bash
python scripts/eval_lora_holdout.py \
  --data_path dataset/lora_medical_eval.jsonl \
  --base_weight full_sft \
  --lora_weight lora_medical_holdout \
  --max_seq_len 768 --batch_size 16 \
  --output out/eval/lora_medical_holdout_metrics_fixed.json
```

## 9. 白盒知识蒸馏

上游默认示例是 MoE teacher 蒸馏 dense student，因此必须先存在匹配形状的 teacher checkpoint，例如 `out/full_sft_768_moe.pth`。下面给出可直接运行的 dense 768 teacher → dense 512 student 链路；先构造学生初始化：

```bash
torchrun --standalone --nproc_per_node 2 train_pretrain.py \
  --data_path ../dataset/pretrain_t2t_mini.jsonl \
  --from_weight none \
  --save_weight pretrain_student \
  --hidden_size 512 \
  --num_hidden_layers 8 \
  --max_seq_len 768 \
  --batch_size 8 \
  --accumulation_steps 2 \
  --dtype bfloat16

torchrun --standalone --nproc_per_node 2 train_full_sft.py \
  --data_path ../dataset/sft_t2t_mini.jsonl \
  --from_weight pretrain_student \
  --save_weight full_sft_student \
  --hidden_size 512 \
  --num_hidden_layers 8 \
  --max_seq_len 768 \
  --batch_size 4 \
  --accumulation_steps 2 \
  --dtype bfloat16
```

再运行蒸馏：

```bash
cd trainer
torchrun --standalone --nproc_per_node 2 train_distillation.py \
  --data_path ../dataset/sft_t2t_mini.jsonl \
  --from_student_weight full_sft_student \
  --from_teacher_weight full_sft \
  --student_hidden_size 512 \
  --student_num_layers 8 \
  --teacher_hidden_size 768 \
  --teacher_num_layers 8 \
  --student_use_moe 0 \
  --teacher_use_moe 0 \
  --save_weight full_dist \
  --alpha 0.5 \
  --temperature 1.5 \
  --batch_size 4 \
  --accumulation_steps 4 \
  --max_seq_len 768 \
  --dtype bfloat16 \
  --epochs 6
```

注意：当前 `init_model` 按 hidden size 查找 `out/<name>_<hidden>.pth`，所以上述学生会从 `out/full_sft_student_512.pth` 加载，教师从 `out/full_sft_768.pth` 加载。Teacher/student 必须共用 tokenizer 和 vocabulary；当前代码只把 teacher logits 截到 student vocab，不解决 token 语义不对齐。若改用 MoE teacher，把 `--teacher_use_moe 1` 打开并先按 6.1 生成 `full_sft_768_moe.pth`。

总损失：`alpha × CE + (1-alpha) × T² KL(teacher || student)`。`T²` 抵消温度对梯度尺度的影响。

### 9.1 本项目的可控 KD A/B 试验

只看 KD 模型自己的 loss 无法证明教师分布有用，因此本项目从同一 `full_sft_student_512` 初始化出发，使用同一个 50,000 条训练子集、样本顺序、seed、optimizer update 数和 effective batch，只改目标函数：

- CE control：`--disable_teacher 1 --alpha 1.0`；
- logit KD：`--disable_teacher 0 --alpha 0.5 --temperature 1.5`。

评测使用独立来源 `rlaif.jsonl` 的 2,000 条 min-hash holdout，不与 50k SFT continuation 共用数据源。可直接使用串行脚本：

```bash
cd /root/minimind
screen -dmS minimind_student_pretrain bash -lc \
  'cd /root/minimind && bash scripts/run_student_pretrain.sh'
screen -dmS minimind_kd_pipeline bash -lc \
  'cd /root/minimind && bash scripts/run_kd_pipeline_after_pretrain.sh'

tail -f out/logs/pretrain_student_512.log
tail -f out/logs/kd_pipeline.log
```

`run_kd_pipeline_after_pretrain.sh` 会依次执行 student SFT、固定子集、CE/KD 两个 continuation 和四 checkpoint 评测，输出 `out/eval/kd_comparison.json`。如果 student SFT 中断且 resume checkpoint 存在，设置 `FROM_RESUME=1 scripts/run_student_sft.sh`；恢复时会恢复 model、optimizer、scaler 和 batch step。

## 10. 可选 DPO 冷启动

在 online RL 前做偏好对齐不是硬性要求，但可以作为对照：

```bash
torchrun --standalone --nproc_per_node 2 train_dpo.py \
  --data_path ../dataset/dpo.jsonl \
  --from_weight full_sft \
  --save_weight dpo \
  --batch_size 2 \
  --accumulation_steps 2 \
  --beta 0.15 \
  --dtype bfloat16
```

DPO 是独立偏好对齐分支，不与四算法串行。为了公平比较 GRPO/CISPO/DAPO/GSPO，四个实验必须从同一个 Agent SFT checkpoint 开始，而不是各自承接前一个算法。

本项目的 DPO 不在原始数据上训完后又回报训练 loss，而是先按 prompt 组固定划分 train/holdout，再评估未见 preference pair。为保证 reference 和 DPO policy 看到完全一致的 chosen/rejected token，评测脚本先物化 `DPODataset`，再计算两模型的 pair margin：

```bash
cd /root/minimind
python scripts/prepare_dpo_data.py

screen -dmS minimind_dpo bash -lc \
  'cd /root/minimind/trainer && torchrun --standalone --nproc_per_node 1 train_dpo.py \
    --data_path ../dataset/dpo_train.jsonl --save_weight dpo_holdout \
    --from_weight full_sft --hidden_size 768 --num_hidden_layers 8 \
    --max_seq_len 768 --batch_size 8 --accumulation_steps 2 \
    --learning_rate 4e-8 --beta 0.15 --dtype bfloat16 --epochs 1'

python scripts/eval_dpo_holdout.py \
  --data_path dataset/dpo_eval.jsonl \
  --reference_weight full_sft --policy_weight dpo_holdout \
  --limit 1000 --output out/eval/dpo_holdout_metrics.json
```

主指标是 chosen preference accuracy、`logp(chosen)-logp(rejected)` pair margin、相对 reference 的 implicit reward margin 和 DPO loss。DPO 与 RLVR 解决的不是同一问题：前者从成对偏好学习，后者从模型自己的 online rollout 和可验证成功信号学习。

实现中最后一个不足 `accumulation_steps` 的窗口会先按实际 remainder 校正梯度、执行 optimizer step，再原子保存模型；不能先保存、循环结束后才更新，否则最终 checkpoint 会漏掉最后几个 micro-batch。

## 11. 先构建 Agentic RL / RLVR 环境

四种策略算法共享同一批轨迹和 verifier，因此必须先让数据、工具执行、action mask 与任务成功定义可信。完整原理和逐函数讲解见 [Agentic RL/RLVR 从零到可运行代码](./02_AGENTIC_RLVR_FROM_ZERO_TO_MASTERY.md)。

数据格式见 `tests/fixtures/agent_rl_smoke.jsonl`。每条样本包含：

- `conversations`：system/user/最终 assistant；训练时 `AgentRLDataset` 去掉最后答案，从剩余消息开始 rollout；
- system 中的 `tools`：提供给 chat template 和 verifier 的 JSON Schema；
- `gt`：可由规则验证的最终事实或数值列表。

多轮执行链：

```text
assistant action
  → <tool_call> JSON 解析与参数校验
  → 环境执行
  → role=tool observation 回写上下文
  → assistant 继续行动或输出最终答案
```

assistant action 与 EOS 进入 policy loss；system、user、tool schema、tool observation 和 PAD 不进入。当前 verifier 分开记录：

- `format_valid_rate`：标签闭合且 JSON 成功解析；
- `tool_call_valid_rate`：工具名在白名单且必填参数合法；
- `tool_execution_success_rate`：模拟工具执行返回成功；
- `tool_evidence_coverage_rate`：执行结果本身覆盖全部 `gt`，防止“调一个合法但无关的工具，然后猜对最终答案”；
- `answer_accuracy`：最终答案覆盖全部 `gt`；
- `task_accuracy`：最终答案、格式、工具要求、执行和轨迹闭合全部成立。

主算法对比使用 `--reward_mode strict`：成功 `+1`、失败 `-1`。`shaped` 只用于格式/工具协议的课程学习，不能与 strict 主表混合。当前工具是 deterministic mock；生产系统必须改为受限进程/容器和不可篡改 execution trace，并增加最终答案与 tool result 的一致性 verifier。

先运行 verifier 单测与小数据 debug：

```bash
cd ..
python -m unittest tests.test_agent_rewards -v
cd trainer
python train_agent.py \
  --from_weight full_sft \
  --data_path ../dataset/agent_rl_math_train.jsonl \
  --save_weight agent_debug \
  --loss_type grpo \
  --batch_size 2 \
  --num_generations 4 \
  --policy_update_epochs 2 \
  --max_gen_len 128 \
  --max_total_len 1024 \
  --reward_mode strict \
  --use_reward_model 0 \
  --debug_mode --debug_interval 1 \
  --metrics_path ../out/metrics/agent_debug.jsonl
```

人工核对至少 20～50 条轨迹，确认工具调用、observation、GT、final answer 和 success 标签一致后，才进入四算法正式训练。

### 11.1 Agent SFT 冷启动：避免严格 RLVR 零梯度

先评估通用 `full_sft`。若同一 prompt 的 G 条采样全部失败，组内标准化 advantage 全为 0，直接上 GRPO/DAPO 不会产生有效策略梯度。本项目使用训练集 oracle 轨迹做小规模协议冷启动，评测样本和标签不参与：

```bash
cd /root/minimind
python scripts/prepare_tool_rlvr_data.py
python scripts/prepare_agent_sft_data.py \
  --tool_train dataset/agent_rl_tool_verified_train.jsonl \
  --math_train dataset/agent_rl_math_train.jsonl \
  --math_limit 512 \
  --output dataset/agent_sft_coldstart.jsonl

python scripts/audit_sft_mask.py \
  --data_path dataset/agent_sft_coldstart.jsonl \
  --max_seq_len 1024 --samples 128 \
  --output out/run_meta/agent_sft_mask_audit_128.json

screen -dmS minimind_agent_sft bash -lc \
  'cd /root/minimind && bash scripts/run_agent_sft.sh'
tail -f out/logs/agent_sft_coldstart.log
```

冷启动前后分别运行 `eval_agent_rlvr.py --num_generations 4`，检查 `dapo_effective_group_rate`。只有出现 `0 < success_count < G` 的混合成败组才进入 RL；若仍为 0，先调整训练集课程难度，不能靠格式加分伪造 success。

本机实跑发现基础 Tool-Use holdout 在 Agent SFT 后达到 100%，已经不适合比较策略优化。为避免在全零 advantage 上“空跑 RL”，正式 Tool-Use 对比改用未进入冷启动标签的组合工具集：

```bash
python scripts/prepare_tool_rlvr_challenge_data.py
python scripts/audit_tool_rlvr_dataset.py \
  --train dataset/agent_rl_tool_challenge_train.jsonl \
  --eval dataset/agent_rl_tool_challenge_eval.jsonl \
  --output out/run_meta/tool_rlvr_challenge_execution_audit.json
```

该集合包含汇率→计算、单位换算→计算、翻译+算术、天气+汇率、时间+单位和三工具组合，共 384 train / 96 holdout，六类严格平衡，问题哈希交集为 0，全量 oracle 重放失败数为 0。Oracle 字段只用于离线审计，`AgentRLDataset` 不读取；`agent_sft_oracle_rows_used=0`。动态采样门禁只在 challenge 的训练提示上校准，正式 96 条 holdout 不参与超参或是否开跑的选择。

## 12. 为什么新增统一策略优化模块

原 `train_grpo.py` 和 `train_agent.py` 的 Torch rollout 由当前模型采样，同时只做一次更新。第一次前向时 `current_logp ≈ old_logp`，ratio 约等于 1，clip 差异很难显现。新增实现固定 rollout 的 `old_logp`，通过 `--policy_update_epochs 2~4` 重用同批轨迹，并在下一批 rollout 前同步新策略。

统一模块支持：

- GRPO：token ratio、对称 clip、先序列 token mean 再样本 mean。
- CISPO：裁剪并 stop-gradient 的 IS weight × advantage × current logp；全局 token mean。
- DAPO：非对称 clip、全局 token mean；Agent 脚本另有动态采样和软超长惩罚。
- GSPO：长度归一化序列 likelihood ratio、序列 clip、序列 reward 与序列 loss。

所有算法均可记录非负 k3 KL，但复现 DAPO、CISPO 论文设置时使用 `--beta 0`，KL 仅作诊断。

训练还会记录 `rollout_logprob_mae` 和 `rollout_ratio_mean`。在每批第一次 policy forward、尚未更新参数时，二者应分别接近 0 和 1。脚本不会在 rollout 后静默左截断超长轨迹，因为截断会改变 action 的条件上下文，使保存的 behavior log-prob 失效；若轨迹超过 `--max_total_len`，训练会明确报出实际最大长度。此时增大 `--max_total_len`，或缩短工具 schema、`--max_turns`、`--max_gen_len` 后从头重跑该 run。

多轮工具轨迹还必须避免 `decode(sampled_ids) → encode(text)` 改写已采 token。当前实现用原始 token ledger 驱动下一轮生成，聊天模板仅用于提取新增 observation 后缀；EOS 保留为 action，PAD 在采样端被抑制。`scripts/audit_multiturn_serialization.py` 会同时覆盖普通可逆案例和当前 tokenizer 的真实非可逆反例。

四种方法的问题链、公式、tensor reduction 和代码分支详见 [策略优化从零到代码精通](./03_POLICY_OPTIMIZATION_FROM_ZERO_TO_MASTERY.md)。

## 13. 单独运行四种算法

以下均在 `trainer/` 执行，使用已经固化的训练划分做可验证数学/工具任务。

共同参数：

```bash
COMMON="--from_weight agent_sft --data_path ../dataset/agent_rl_math_train.jsonl --epochs 1 --batch_size 2 --num_generations 8 --policy_update_epochs 4 --beta 0 --dtype bfloat16 --reward_mode strict --use_reward_model 0 --max_gen_len 768 --max_total_len 2500"
```

GRPO：

```bash
torchrun --standalone --nproc_per_node 1 train_agent.py $COMMON \
  --loss_type grpo \
  --save_weight agent_grpo_s42 \
  --seed 42 \
  --metrics_path ../out/metrics/grpo_s42.jsonl
```

CISPO：

```bash
torchrun --standalone --nproc_per_node 1 train_agent.py $COMMON \
  --loss_type cispo \
  --epsilon_high 5.0 \
  --save_weight agent_cispo_s42 \
  --seed 42 \
  --metrics_path ../out/metrics/cispo_s42.jsonl
```

这里 `epsilon_high=5.0` 是论文的 `ε_high^IS`，实际 IS ratio 上界为 `1+ε=6`；不是把 ratio 直接裁到 5。

MiniMax-M1 完整 recipe 还复用 DAPO 的动态采样和长度惩罚。主对照为隔离 loss 不打开；单独复现组合时运行：

```bash
torchrun --standalone --nproc_per_node 1 train_agent.py $COMMON \
  --loss_type cispo \
  --epsilon_high 5.0 \
  --dynamic_sampling \
  --dynamic_sampling_rounds 10 \
  --overlong_cache_len 128 \
  --save_weight agent_cispo_ds_s42 \
  --seed 42 \
  --metrics_path ../out/metrics/cispo_ds_s42.jsonl
```

该结果应标为 `CISPO + Dynamic Sampling + Length`，不能把组合收益全部归因于 CISPO objective。

DAPO：

```bash
torchrun --standalone --nproc_per_node 1 train_agent.py $COMMON \
  --loss_type dapo \
  --dapo_epsilon_low 0.2 \
  --dapo_epsilon_high 0.28 \
  --dynamic_sampling \
  --dynamic_sampling_rounds 10 \
  --overlong_cache_len 128 \
  --overlong_penalty_coef 1.0 \
  --save_weight agent_dapo_s42 \
  --seed 42 \
  --metrics_path ../out/metrics/dapo_s42.jsonl
```

GSPO：

```bash
torchrun --standalone --nproc_per_node 1 train_agent.py $COMMON \
  --loss_type gspo \
  --gspo_epsilon_low 0.0003 \
  --gspo_epsilon_high 0.0004 \
  --save_weight agent_gspo_s42 \
  --seed 42 \
  --metrics_path ../out/metrics/gspo_s42.jsonl
```

说明：GSPO 的 `3e-4/4e-4` 来自论文 30B MoE 实验，并不保证对 64M 模型最优。先用它复现公式，再在独立验证集调参；不能看测试集挑 clip。

DAPO 动态采样保留每组 `0 < success_count < G` 的 prompt，重复采候选直到有效 batch 和基线相同。若 10 轮仍填不满，脚本会报错，而不是悄悄减少更新数。此时应检查 verifier、缩短任务难度跨度或增加轮数，不能把稠密格式 reward 当 success。

## 14. 一键三随机种子正式实验

从仓库根目录：

```bash
MM_GPUS=1 \
MM_SEEDS="42 43 44" \
MM_BATCH_SIZE=2 \
MM_GENERATIONS=8 \
MM_POLICY_EPOCHS=4 \
bash scripts/run_rlvr_ablation.sh
```

脚本会为四算法分别从同一 `agent_sft` 初始化，输出独立 checkpoint 与 JSONL，并生成：

```text
out/metrics/algorithm_comparison.csv
```

公平性同时报告两种预算：相同 optimizer update 数和实际 candidate prompt/rollout token 数。DAPO 动态采样通常消耗更多候选，不能只按 step 声称更高效率。

## 15. 固定集评测：产生真正可写入简历的数字

训练完成后从根目录运行：

```bash
python scripts/eval_agent_rlvr.py \
  --weights agent_sft,agent_math_grpo_s42,agent_math_grpo_s43,agent_math_grpo_s44,agent_math_cispo_s42,agent_math_cispo_s43,agent_math_cispo_s44,agent_math_dapo_s42,agent_math_dapo_s43,agent_math_dapo_s44,agent_math_gspo_s42,agent_math_gspo_s43,agent_math_gspo_s44 \
  --reference_weight agent_sft \
  --data_path ./dataset/agent_rl_math_eval.jsonl \
  --seeds 42,43,44 \
  --limit 256 \
  --batch_size 2 \
  --num_generations 1 \
  --thinking_ratio 0 \
  --output_dir ./out/eval_rlvr
```

再按“checkpoint 内先平均解码 seed，算法内再跨训练 seed”聚合：

```bash
python scripts/summarize_eval_results.py \
  out/eval_rlvr/summary.csv \
  --output out/eval_rlvr/algorithm_comparison.csv
```

输出：

- `summary.csv`：每 checkpoint/seed 的 Reward、任务准确率、答案准确率、KL、平均/P95 长度、unfinished rate、吞吐。
- `trajectories.jsonl`：每个样本的 GT、完整输出和所有 verifier 指标，可抽查误判。

评测 seed 控制采样，但 checkpoint seed 才代表训练方差；不能把 3 个解码 seed 当成 3 次独立训练。建议报告跨训练 seed 的均值±标准差，并对题目级 paired difference 做 95% bootstrap 置信区间。

### 15.1 本机实际使用的 Formal 与最终验收命令

完成 Agent SFT、DPO/Pilot 与架构验证后，本机用以下两个独立 `screen` 会话串起两任务 Formal 和最终门禁：

```bash
cd /root/minimind
screen -dmS minimind_rlvr_formal bash -lc \
  'cd /root/minimind && bash scripts/run_rlvr_formal_after_architecture.sh'

screen -dmS minimind_finalize bash -lc \
  'cd /root/minimind && bash scripts/run_finalize_after_formal.sh'

tail -f out/logs/rlvr_formal_pipeline.log
# 另一个终端观察最终门禁：
tail -f out/logs/finalize_pipeline.log
```

Formal 脚本会依次运行 math/tool；每任务 4 算法 × seeds 42/43/44、每 run 最多 10 个 accepted groups、G=4、policy epochs=2、LR=`1e-7`，然后对共同 Agent SFT baseline 与 12 个 checkpoint 运行 64 prompts × decode seeds 101/102/103。DAPO 的动态采样轮数为 50，确保稀疏 Tool 任务有足够候选预算。

最终验收会等待 Formal 成功 marker，再顺序执行 Agent SFT mask、MoE 路由、实际模型 Flash profiler、多轮序列化、两套工具 oracle、`compileall`、31 项单测、checksum 与证据索引。判断完成不能只看 screen 消失，应执行：

```bash
grep -q 'RLVR formal pipeline complete' out/logs/rlvr_formal_pipeline.log
grep -q 'final validation complete' out/logs/finalize_pipeline.log
python - <<'PY'
import json
e = json.load(open('out/run_meta/experiment_evidence.json'))
assert all(e['pipeline_markers'].values()), e['pipeline_markers']
assert e['missing'] == [], e['missing']
print('all pipeline markers true; missing=[]')
PY
```

最终主要产物为 `out/metrics/algorithm_comparison_{math,tool}.csv`、`out/eval_rlvr/{math,tool}/{summary.csv,algorithm_comparison.csv,trajectories.jsonl}` 和 `out/run_meta/experiment_evidence.json`。

## 16. 可选 SGLang 训推分离

先把 PyTorch checkpoint 转为 Transformers 格式，具体参数看官方 `scripts/convert_model.py --help`。在独立 GPU/终端启动：

```bash
python -m sglang.launch_server \
  --model-path ./minimind-3 \
  --attention-backend triton \
  --host 127.0.0.1 \
  --port 8998
```

训练终端：

```bash
cd trainer
python train_agent.py \
  --rollout_engine sglang \
  --sglang_base_url http://127.0.0.1:8998 \
  --sglang_model_path ../model \
  --sglang_shared_path ./sglang_ckpt_agent \
  --data_path ../dataset/agent_rl_math_train.jsonl \
  --loss_type gspo \
  --policy_update_epochs 4 \
  --rollout_sync_interval 1
```

`rollout_sync_interval=1` 是算法对比的正确性优先设置：每批更新后把新策略同步到推理端。增大间隔会提高吞吐但增加 policy lag，必须记录 old/current policy version 和 ratio/KL，否则比较失真。

## 17. 指标判读与停止规则

至少同时看：

- Reward 与验证集 `task_accuracy`：训练 reward 上升而验证准确率不升，通常是过拟合或 reward hacking。
- `kl_k3`：突然飙升说明策略偏离 reference；长期接近 0 可能更新无效。
- `ratio_mean/std`、`clip_fraction`：ratio 恒为 1 表明没有真正 off-policy 的多轮更新；clip 过高表示步长或 policy epoch 太大。
- `group_reward_std`、`zero_variance_group_rate`：退化组多时 GRPO 信号不足。
- 平均与 P95 响应长度、unfinished rate：只看平均长度会隐藏长尾爆炸。
- DAPO acceptance：过低说明任务难度与当前模型能力不匹配，且 rollout 成本上升。
- MoE aux loss 与专家负载：本次日志有 aux loss；如正式做 MoE，应再增加每专家 token 比例和路由切换率。

停止条件示例：连续 3 次固定集评测准确率下降；KL 或长度超过预设阈值；NaN/Inf；tool execution rate 下降而 reward 上升；三 seed 只有一个 seed 改善。

## 18. 常见故障

OOM：先降低 `max_gen_len`、`num_generations`、`batch_size`，再增加 accumulation。RL 同时驻留 policy/reference，显存比 SFT 高；SGLang 最好使用独立 GPU。

Reward 全相同：检查 `gt` 类型、最终答案解析、工具是否真的执行；打开 `--debug_mode --debug_interval 1` 抽样。

DAPO 填不满：增加 `--dynamic_sampling_rounds` 只是最后手段；先确认任务不是全会或全不会，以及二值 verifier 没有误杀。

GSPO 全部被 clip：小模型上调大序列 clip，但必须在验证集调。先确认 `old_logp` 对齐 response token，tool observation mask 为 0。

SGLang ratio 异常：检查权重同步成功、tokenizer 相同、训练/推理 kernel 精度差；用 Torch rollout 作为 correctness oracle。

DDP hang：所有 rank 必须执行相同数量的 forward/backward 和 collective。动态采样使用全局 ready 的 MIN all-reduce，不能在单个 rank 私自进入更新。

FP16 NaN：Pretrain/SFT/LoRA/KD 有 GradScaler；新增 RL 建议 BF16。若硬件不支持 BF16，再为 RL 接入 GradScaler，不要只把参数改成 `float16`。

## 19. 最终验收清单

```bash
python -m unittest discover -s tests -v
python -m compileall model dataset trainer scripts
test -f out/pretrain_768.pth
test -f out/full_sft_768.pth
test -f out/metrics/algorithm_comparison_math.csv
test -f out/metrics/algorithm_comparison_tool.csv
test -f out/eval_rlvr/math/algorithm_comparison.csv
test -f out/eval_rlvr/tool/algorithm_comparison.csv
test -f out/run_meta/experiment_evidence.json
grep -q 'final validation complete' out/logs/finalize_pipeline.log
```

最终归档：代码 commit、环境快照、训练配置、每步 JSONL、可选的外部平台导出、全部 checkpoint、固定评测 manifest、逐样本轨迹、汇总 CSV、checksum 和失败实验。只有这些材料齐全，简历中的量化结果才可被复核。
