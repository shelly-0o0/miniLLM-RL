# Stage 1：MiniMind-64M GSM8K Agentic RL 最终报告

完成日期：2026-10-05
代码基线提交：`47be6fab0c03eac57c2577a9bd20ff0408b85b61`
正式比较范围：Agent-SFT、GRPO、CISPO、DAPO、GSPO
最终选择：**DAPO**

> 本报告只总结新的 GSM8K Agentic RL 主线。仓库中旧的 math/tool 小规模实验不是本轮正式结果。项目计划曾列出 PPO，但本轮锁定并完成的正式矩阵是四个 group-relative 方法；因此本文不声称得到 PPO 对比结论。

## 1. 最终结论

在 747 道固定 validation 上，DAPO 的三个独立训练种子取得最高的严格任务准确率：

- Agent-SFT：`1.9188%`
- DAPO：`2.9154% ± 0.3847 pp`（训练种子样本标准差）
- 绝对增量：`+0.9966 pp`
- 题目级 paired bootstrap 95% CI：`[+0.2826, +1.6808] pp`

根据预先约定的 validation 主指标选择 DAPO 后，只对 Agent-SFT 和 DAPO 执行一次正式 test 流程。完整 1,319 道 GSM8K official test 的结果为：

- Agent-SFT：`2.1986%`
- DAPO：`3.3190% ± 0.3047 pp`
- 绝对增量：`+1.1204 pp`
- 相对增量：约 `+50.96%`
- 题目级 paired bootstrap 95% CI：`[+0.5391, +1.7185] pp`

DAPO 的三个训练种子在 test 上分别为 `3.2853%`、`3.6391%`、`3.0326%`，都高于同一 Agent-SFT 基线的 `2.1986%`。这说明改善不是单个训练种子的偶然尖峰；但独立训练种子只有 3 个，仍不足以作强泛化或 SOTA 声明。

最重要的限制是绝对准确率仍然很低。模型已经几乎总能输出合法工具调用并成功执行 calculator，但大约 96% 的轨迹仍选择了错误表达式或错误答案。因此本轮证明的是“完整 Agentic RL 闭环和 DAPO 增量有效”，不是“64M 模型已经解决 GSM8K”。

## 2. 实验问题与数据边界

本轮回答三个问题：

1. 只使用 train 的 oracle 轨迹做 Agent-SFT，能否建立可训练的工具调用冷启动？
2. 在共同 Agent-SFT 起点、共同候选 rollout 预算下，四种 group-relative 策略目标谁在 validation 上最好？
3. validation 选出的算法能否在从未用于选择的 official test 上保持增量？

固定数据如下：

| Split | 行数 | 用途 | SHA-256 |
|---|---:|---|---|
| `train_rl.jsonl` | 6,726 | RL prompt | `ec9aecf7083386226e17f9eead206b4e045e952ac2c28080f9c9120da74fbf3e` |
| `train_sft.jsonl` | 6,637 | Agent-SFT oracle 轨迹 | `132a54eb2266ac75ea44c4055ca6ce3661df5b80aa556ff1b602366cdbf772cb` |
| `validation_rl.jsonl` | 747 | 选算法，不更新参数 | `a7f6c3d91f63269ab31be576d352ccbb19e6280f1fb3818e2eb4d78b8be13700` |
| `test_rl.jsonl` | 1,319 | 选择结束后的最终报告 | `909ce396576b5cfe2c64889c0e2159632d532b77b8854740e951aa06e41b7ca9` |

官方 train 的 7,473 题按固定种子题目级拆成 6,726 train 和 747 validation；官方 test 的 1,319 题保持不变。三者 ID 互斥。`train_sft` 比 `train_rl` 少 89 行，是因为这些题没有可回放的 calculator annotation；它们仍可进入 RL，但不能伪造为工具监督标签。

validation/test 中虽然保留 `gt` 和 oracle 字段供 verifier 与审计使用，rollout prompt 只包含 system、user 和空 assistant sentinel，gold 不进入模型输入。

## 3. 从数据到最终测试的完整流程

```text
GSM8K official train/test
  → 固定 train/validation/test + manifest + 泄漏检查
  → 安全回放 <<expression=result>> calculator annotations
  → train_sft / train_rl / validation_rl / test_rl
  → assistant-only mask 审计
  → MiniMind-64M Agent-SFT
  → 同一 Agent-SFT 独立分叉 4 algorithms × 3 training seeds
  → 固定 validation：每 checkpoint × 3 decode seeds
  → 先合并 decode seeds，再跨 training seeds 聚合
  → validation 选择 DAPO
  → Agent-SFT + DAPO(3 training seeds) official test
  → paired bootstrap、失败归因、权重与产物哈希审计
```

### 3.1 环境与回归测试

- PyTorch：`2.6.0+cu124`
- torchvision：`0.21.0+cu124`
- CUDA 12.4 wheels
- 模型：MiniMind Dense，hidden size 768、8 层、非 MoE
- 参数日志：约 `63.91M`
- 最终全仓回归：`67/67 PASS`

最初 `requirements.txt` 报不存在，是因为命令在 home 目录执行，而不是仓库根目录；这不是依赖文件损坏。最终回归首次在受限本地环境中还遇到 Hugging Face cache lock 的只读文件系统错误，改用可写的 `HF_HOME=/tmp/mini-rl-hf-stage1-final` 后 67 项全部通过，确认是运行环境路径问题而不是测试失败。

### 3.2 数据生成与监督 mask

核心命令：

```bash
python scripts/download/download_gsm8k.py
python scripts/prepare/prepare_gsm8k.py
python scripts/prepare/prepare_gsm8k_agent_data.py

python scripts/audit_sft_mask.py \
  --data_path data/processed/gsm8k_agent/train_sft.jsonl \
  --tokenizer_path models/minimind-3 \
  --max_seq_len 1024 \
  --samples 128 \
  --examples 3 \
  --seed 42 \
  --output out/run_meta/gsm8k_agent_sft_mask_audit.json
```

mask 审计结果：

- 采样：128 条
- assistant supervised tokens：16,724
- non-pad tokens：68,921
- assistant token ratio：24.2655%
- mismatch：0
- zero-supervision：0

早期 `datasets.CastError` 来自 JSONL 顶层新增 `id`，而旧 `features` schema 只声明了 `conversations`。最终 loader 让 datasets 从 JSON 自动推断顶层列，同时由训练代码只读取需要的字段；对应回归测试覆盖顶层审计 metadata。

### 3.3 Agent-SFT

本轮等价训练配置如下：

```bash
python -u trainer/train_full_sft.py \
  --data_path data/processed/gsm8k_agent/train_sft.jsonl \
  --tokenizer_path models/minimind-3 \
  --from_weight full_sft \
  --from_save_dir out \
  --save_dir out \
  --save_weight agent_sft \
  --hidden_size 768 \
  --num_hidden_layers 8 \
  --use_moe 0 \
  --max_seq_len 1024 \
  --batch_size 4 \
  --accumulation_steps 4 \
  --epochs 3 \
  --learning_rate 1e-5 \
  --dtype bfloat16
```

`4 × 4 = 16` 是有效 batch 的说明，不是 shell 命令。三轮各 1,660 micro-batches；训练 loss 从早期约 0.2–0.3 降到后期约 0.08–0.17。mask 只监督 assistant tool-call 和 final-answer token，不把 user/tool observation 当语言模型目标。

正式 Agent-SFT 权重审计：

- 91 tensors，90 个 tensor 相对 Full-SFT 改变
- 97.4728% 的已保存参数元素发生改变
- relative L2 change：`0.0113506`
- maximum absolute change：`0.00463867`
- non-finite tensors：0
- checkpoint SHA-256：`46741bb4ce8ec4f08a15a1f811a734f59c2df3054a42dce2a0e2c2097343f207`

### 3.4 正式 RL 协议

每个算法和训练种子都从同一个 `out/agent_sft_768.pth` 独立加载，不串行继承其他算法：

```bash
for algorithm in grpo cispo dapo gspo; do
  for seed in 42 43 44; do
    bash scripts/run_gsm8k_rl_full.sh "$algorithm" "$seed"
  done
done
```

锁定的共同超参数：

| 项目 | 值 |
|---|---:|
| training prompts | 6,726 |
| candidate groups/run | 6,726 |
| trajectories/group | 8 |
| candidate trajectories/run | 53,808 |
| policy update epochs | 2 |
| learning rate | `3e-7` |
| weight decay | 0 |
| max turns | 3 |
| max generation length | 384 |
| max total length | 2,500 |
| reward | strict deterministic RLVR，成功 `+1`，失败 `-1` |
| sampling | temperature 1.0、top-k 0、top-p 1.0 |
| checkpoint dtype | FP32 |

严格成功同时要求：轨迹完成、数值答案正确、必需工具被合法执行、最终答案能够由结构化工具结果支撑。只写对答案、格式正确但未执行 calculator、执行错误表达式后猜答案，都不能成功。

### 3.5 候选预算与有效组

公平性锁定在**相同候选 rollout 预算**，不是强行让每个算法拥有相同有效组。DAPO 的 dynamic sampling 会丢弃组内奖励完全相同的 prompt group；其他算法仍处理这些组，但归一化 advantage 为零时没有有效学习信号。

| 算法 | 3 seeds candidate groups | candidate trajectories | effective groups | effective rate | optimizer updates | generated tokens | 单卡 wall time 合计 |
|---|---:|---:|---:|---:|---:|---:|---:|
| GRPO | 20,178 | 161,424 | 20,178 | 100% nominal | 40,356 | 19,940,206 | 26.69 h |
| CISPO | 20,178 | 161,424 | 20,178 | 100% nominal | 40,356 | 19,722,940 | 33.22 h |
| DAPO | 20,178 | 161,424 | 3,745 | 18.56% | 7,490 | 18,196,702 | 26.65 h |
| GSPO | 20,178 | 161,424 | 20,178 | 100% nominal | 40,356 | 19,940,733 | 30.71 h |

总计：80,712 candidate groups、645,696 rollout trajectories、77,800,581 generated tokens、约 117.26 单卡 GPU-hours。DAPO 三个种子的有效组分别为 1,265、1,245、1,235。

所有 12 个 run 都满足：

- `budget_complete=true`
- `candidate_groups=6726`
- `candidate_trajectories=53808`
- 91/91 tensors 相对 Agent-SFT 改变
- checkpoint 无非有限值
- 每个 checkpoint 均记录独立 SHA-256

## 4. Validation 统计与选模

每个 checkpoint 在完全相同的 747 题上使用 decode seeds 42/43/44。统计时先对同一 checkpoint 的三个 decode seed 取均值，再把三个训练 checkpoint 当作独立训练 run 计算均值和样本标准差；不能把 decode seed 当成三个独立训练实验。

| 模型/算法 | training runs | Strict Task Acc | Answer Acc | Reward | Evidence Coverage | Avg Length | KL k3 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Full-SFT | 1 | 0.2231% | 1.2941% | -0.99554 | 1.2048% | 149.09 | 0（vs Full-SFT） |
| Agent-SFT | 1 | 1.9188% | 2.0973% | -0.96162 | 1.9634% | 123.91 | 0.42208（vs Full-SFT） |
| GSPO | 3 | 2.2906% ± 0.2690 pp | 2.5584% | -0.95419 | 2.4096% | 125.36 | 0.000692（vs Agent-SFT） |
| CISPO | 3 | 2.3948% ± 0.2201 pp | 2.7369% | -0.95210 | 2.4840% | 123.59 | 0.000993（vs Agent-SFT） |
| GRPO | 3 | 2.4691% ± 0.4000 pp | 2.7220% | -0.95062 | 2.5138% | 125.75 | 0.000984（vs Agent-SFT） |
| **DAPO** | **3** | **2.9154% ± 0.3847 pp** | **3.1980%** | **-0.94169** | **2.9451%** | **114.10** | **0.014320（vs Agent-SFT）** |

不同 reference 的 KL 不能相减：早期 Full-SFT/Agent-SFT 文件以 Full-SFT 为 reference，四个 RL 算法以 Agent-SFT 为 reference。最终 CSV 显式记录 `kl_reference`，不再生成误导性的跨 reference KL delta。

相对 Agent-SFT 的题目级 paired bootstrap：

| 算法 | Task Acc Δ | 95% CI | P(Δ>0) |
|---|---:|---:|---:|
| CISPO | +0.4760 pp | [-0.1934, +1.1304] pp | 0.9201 |
| GRPO | +0.5503 pp | [-0.1487, +1.2494] pp | 0.9386 |
| GSPO | +0.3719 pp | [-0.3570, +1.0858] pp | 0.8385 |
| **DAPO** | **+0.9966 pp** | **[+0.2826, +1.6808] pp** | **0.9963** |

因此按预注册主指标 `strict task_accuracy` 选择 DAPO。test 在这一步之前没有用于调参、换算法或挑训练种子。

## 5. Official test 最终结果

测试命令被固化为：

```bash
bash scripts/run_gsm8k_stage1_final_test.sh agent_sft
bash scripts/run_gsm8k_stage1_final_test.sh dapo 42
bash scripts/run_gsm8k_stage1_final_test.sh dapo 43
bash scripts/run_gsm8k_stage1_final_test.sh dapo 44
```

每个 checkpoint 都跑完整 1,319 题 × decode seeds 42/43/44。Agent-SFT 有 3,957 条测试轨迹，DAPO 三训练种子合计 11,871 条。

| 指标 | Agent-SFT | DAPO（3 training seeds） | Δ |
|---|---:|---:|---:|
| Strict Task Acc | 2.1986% | **3.3190% ± 0.3047 pp** | **+1.1204 pp** |
| Answer Acc | 2.5272% | **3.6475%** | **+1.1204 pp** |
| Reward | -0.95603 | **-0.93362** | +0.02241 |
| Format Valid | **99.7220%** | 99.6967% | -0.0253 pp |
| Tool Call Valid | **99.9874%** | 99.9836% | -0.0038 pp |
| Tool Execution Success | 99.7914% | **99.8874%** | +0.0960 pp |
| Required Tool Coverage | **100.0000%** | 99.9916% | -0.0084 pp |
| Evidence Coverage | 2.2492% | **3.3274%** | **+1.0783 pp** |
| Avg Response Length | 126.19 | **114.72** | -11.47 tokens |
| KL k3 vs Agent-SFT | 0 | 0.014527 ± 0.001306 | +0.014527 |

DAPO 分训练种子：

| Train seed | Strict Task Acc | Answer Acc |
|---:|---:|---:|
| 42 | 3.2853% | 3.6644% |
| 43 | 3.6391% | 4.0182% |
| 44 | 3.0326% | 3.2600% |

test 的题目级 paired bootstrap：

- Δ：`+1.1204 pp`
- 95% CI：`[+0.5391, +1.7185] pp`
- bootstrap 中 Δ>0 的比例：`0.9999`
- 题目层面 DAPO win/tie/loss：`15.47% / 79.30% / 5.23%`

该 bootstrap 以题目为重采样单位，描述题目抽样不确定性；它不能替代更多训练种子。训练随机性由三 DAPO checkpoint 的 `0.3047 pp` 样本标准差单独呈现。

## 6. 失败归因

test 轨迹的首要失败原因：

| 算法 | Success | Answer incorrect | Evidence not grounded | Format invalid | Unfinished |
|---|---:|---:|---:|---:|---:|
| Agent-SFT | 2.199% | 97.195% | 0.329% | 0.253% | 0.025% |
| DAPO | 3.319% | 96.058% | 0.312% | 0.253% | 0.051% |

工具调用和执行率已接近 100%，所以主要瓶颈不是 XML/JSON 外形，也不是 calculator runtime，而是：

1. 从文字题构造正确算式；
2. 多步中间结果的组合；
3. 最终数字与实际执行证据一致。

DAPO 的收益主要来自正确答案和证据覆盖同时提高，而不是靠格式奖励“刷分”。格式率甚至轻微下降 `0.0253 pp`，严格成功仍上升 `1.1204 pp`。

## 7. 训练过程中发现并保留的异常

### 7.1 SFT schema mismatch

问题：`train_sft.jsonl` 顶层包含 `id`，旧固定 feature schema 不允许额外列，导致 `datasets.CastError`。
处理：允许 JSON schema 推断；训练只读取需要的 conversations；新增回归测试。
结果：mask audit mismatch=0、zero-supervision=0。

### 7.2 DAPO malformed tool schema crash

问题：模型偶发生成 dict 类型 tool name，旧奖励路径把它当作 hash key，触发 `TypeError: unhashable type: 'dict'`。
处理：tool name 必须是 string、arguments 必须是 dict，否则记为无效调用而不是让训练进程崩溃；新增 `test_malformed_tool_schema_is_rejected_without_crashing`。

### 7.3 DAPO resume 预算不公平

问题：早期 resume 段重置 candidate counter，使重跑后的 DAPO seed 43/44 超过正式的 6,726 candidate groups。
处理：不把它们包装成有效结果，保留式归档并从共同 Agent-SFT 重新训练：

- `seed_43.invalid_crash_resume_budget_20261005`
- `seed_44.invalid_resume_budget_20261005`

最终进入统计的 seed 43/44 都是 fresh run，且 candidate groups 恰好为 6,726。

### 7.4 Pilot 与正式实验的边界

10-update smoke/pilot 只验证权重能改变、loss/KL 有限和 checkpoint 可保存。它们没有足够预算，不能替代正式 6,726-group 训练，也不进入最终表格。

## 8. 可复现统计命令

```bash
HF_HOME=/tmp/mini-rl-hf-stage1-final PYTHONPATH=. \
python -m unittest discover -s tests -p 'test_*.py' -v

python scripts/analyze_gsm8k_stage1.py \
  --include-test \
  --bootstrap-samples 20000
```

分析器会执行以下硬检查：

- 12 个 formal run 都完成预算；
- 每个 run 恰好 6,726 candidate groups / 53,808 candidate trajectories；
- validation 每 checkpoint 恰好 747 × 3 条轨迹；
- test 每 checkpoint 恰好 1,319 × 3 条轨迹；
- decode seeds 恰好为 42/43/44，dataset index 无重复或缺失；
- 12 个 checkpoint key 与 Agent-SFT 一致；
- 所有 checkpoint tensor finite；
- 统计顺序是 decode seed → checkpoint → training seed；
- winner 只从 validation 选择；
- 20,000 次 paired bootstrap 使用固定 seed 20261005。

## 9. 机器可读证据

最终目录：`out/stage1_final/`

| 文件 | 内容 |
|---|---|
| `training_runs.csv` | 12 个训练 run 的预算、耗时、token、checkpoint hash 与参数变化审计 |
| `validation_by_checkpoint.csv` | 逐 checkpoint 的三 decode-seed 均值 |
| `validation_by_algorithm.csv` | 先 checkpoint 后 training-seed 的正式聚合 |
| `validation_paired_bootstrap.csv` | 四算法相对 Agent-SFT 的题目级配对区间 |
| `test_by_checkpoint.csv` | Agent-SFT 与三个 DAPO checkpoint 的 test 结果 |
| `test_by_algorithm.csv` | 最终 test 聚合 |
| `test_paired_bootstrap.csv` | DAPO 相对 Agent-SFT 的 test 配对区间 |
| `failure_modes.csv` | validation/test 轨迹失败类型 |
| `stage1_evidence.json` | 协议、winner、计数及数据/代码/权重 SHA-256 |
| `generated_artifacts.sha256` | 上述生成文件的 SHA-256 |

原始证据仍保留在：

- `out/metrics/gsm8k_{algorithm}_full_s{seed}.jsonl`
- `out/rl_full/{algorithm}/seed_{seed}/`
- `out/eval_full/`
- `out/eval_test/stage1/`
- `out/logs/`
- `out/run_meta/`

## 10. 结论边界与后续建议

可以据实声称：

1. 完成了 64M 模型的多轮 calculator Agent-SFT → RLVR → fixed validation → untouched official test 闭环；
2. 四算法、三训练种子在相同 candidate rollout 预算下完成；
3. DAPO 在 validation 排名第一，并在 official test 保持约 `+1.12 pp` 的严格准确率增量；
4. 增量来自正确答案/证据覆盖，而不是伪造格式成功；
5. 所有正式权重 finite、预算与哈希可审计。

不能声称：

1. 达到 GSM8K SOTA 或具备高数学能力；
2. paired item bootstrap 已完全覆盖训练随机性；
3. 本轮对 PPO 得出了结论；
4. tool execution 接近 100% 等于题目求解正确；
5. test 可以继续用于超参数搜索。

如果继续优化 Stage 1，应重新划分新的调参 validation 或采用嵌套验证，优先改进算式规划和中间证据质量；official test 应保持冻结，不再用它挑学习率、clip 或 curriculum。
