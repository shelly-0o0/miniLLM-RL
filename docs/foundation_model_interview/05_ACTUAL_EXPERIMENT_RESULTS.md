# RTX 4090 真实实验结果与面试口径

> 更新时间：2026-08-30。本文只记录该服务器真实生成并已归档的产物。实验基线为 MiniMind commit `393e387e9ad99f0f04c296e4c5e7353f4444629f`，单张 NVIDIA RTX 4090 23.52 GiB，PyTorch 2.6.0+cu124，BF16。

> 2026-09-01 补充：本文保留初始 Formal 的负结果作为审计基线。FP32 checkpoint、行为 log-prob 对齐、Tool 课程冷启动和后续严格 DAPO 的新增结果见 [RL 不增长诊断与后续优化报告](./09_RL_OPTIMIZATION_FOLLOWUP.md)。

## 1. 最终完成状态

| 阶段 | 状态 | 可用证据 |
|---|---|---|
| Dense Pretrain 768/8 | 已完成 | 2 epochs，79,390 batch/epoch，最后一批 loss 1.8342 |
| Dense SFT 768/8 | 已完成 | 2 epochs，退出码 0，10,408 s，最后一批 loss 1.4291 |
| Assistant-only mask | 已完成 | 128 样本，mismatch=0，zero-supervision=0 |
| LoRA 医疗适配 | 已完成 | 参数 0.393M / 0.61%；独立 holdout 见第 3 节 |
| 稀疏 MoE | 短程机制验证 | 训练到 8,350/79,390，不做质量超过 Dense 的声明 |
| KD | 已完成 | 50k 同预算 CE/KD 对照 + 2,000 条独立 RLAIF holdout，见第 4 节 |
| DPO | 已完成 | 1,000 对独立 holdout；无提升，见第 5 节 |
| Agent SFT 与 RLVR readiness | 已完成 | 冷启动前后固定集、基础工具饱和诊断和组合挑战集，见第 8 节 |
| 架构验证 | 已完成 | Flash profiler、4096-token RoPE/YaRN、MoE 固定批审计，见第 5–6 节 |
| 四算法 Pilot | 已完成 | 两任务各 4 算法 × 1 seed × 3 updates；固定集与 SFT 持平 |
| 四算法 Formal | 已完成 | 两任务各 4 算法 × 3 训练 seed；共同 baseline + 12 checkpoints，各 64 prompts × 3 decode seeds |
| 最终验收与证据索引 | 已完成 | 31/31 tests；7 个 pipeline markers 全 true；`missing=[]` |

## 2. Dense 主线与训练稳定性

Dense 主线是 63,912,192 参数的 Decoder-only 模型，hidden size 768、8 层、8 个 Query heads、4 个 KV heads。Pretrain 使用 sequence length 768、micro-batch 16、梯度累积 4、BF16、学习率 `5e-4`、两轮；SFT 使用同一架构和 Assistant-only loss mask。

SFT mask 审计在 128 个固定样本上统计到：70,552 个 supervised tokens、83,473 个 non-pad tokens，监督比例 84.52%；逐 token 预期 mask 与实际 labels 的 mismatch 为 0，没有零监督样本。这证明 user/system prompt 没有被错误算入 SFT loss。

面试时不要只报“最后一批 loss”当作泛化能力；它只是训练动态证据，而非未见集评测。

复盘还发现上游保存顺序的一个边界问题：当 epoch 的 micro-batch 数不能被梯度累积步数整除时，旧脚本在最后一个 remainder optimizer step 之前保存。现有 Dense Pretrain 因而少持久化最后 1/39,696 个预期 optimizer updates（约 0.0025%），学生 Pretrain 少 1/19,848（约 0.0050%）；日志中的最终 forward loss 仍存在，但对应更新不在权重里。该问题已统一修成“按 remainder 校正梯度 → optimizer step → 原子保存”，SFT/Agent SFT 的本次 batch 数可整除而不受影响。不要把这个极小差异描述成性能提升，但面试中应能说明如何定位和修复。

## 3. LoRA 的独立 holdout 结果

LoRA 从 `full_sft_768` 分支，只训练 393,216 个 adapter 参数，占完整参数约 0.61%。为避免用训练 loss 冒充结果，医疗数据按固定划分得到 22,792 条训练和 2,484 条 holdout，并先物化同一批 token，确保 base 与 LoRA 的评测输入完全一致。

| 模型 | Assistant tokens | Token-weighted loss | PPL |
|---|---:|---:|---:|
| Full SFT base | 785,657 | 1.698209 | 5.464150 |
| + LoRA medical | 785,657 | 1.568724 | 4.800518 |

LoRA 相对 base 的 holdout loss 下降 7.62%，PPL 下降 12.15%。这个结论只适用于该医疗 holdout，不能推广为通用能力提升；本试验也没有做医疗事实性/安全性评估。

该已评测 LoRA checkpoint 同样来自保存顺序修复前：optimizer state 显示持久化了 2,138/2,139 个预期更新，少最后一个 remainder update（约 0.047%）。上表就是这个实际文件的结果，不对缺失更新作任何收益推断；后续 LoRA 运行已使用修复后的顺序。

## 4. 知识蒸馏的独立 holdout 结果

学生为 hidden size 512、8 层、30,025,216 参数，教师为 768/8、63,912,192 参数；学生参数减少 53.02%。CE 与 KD 都从同一个 `full_sft_student_512` 出发，使用同一 50,000 条训练子集、相同顺序/seed/effective batch/6,250 micro-batches；KD 唯一改变是启用 768 教师并使用 `0.5×CE + 0.5×T²KL`、`T=1.5`。训练集来自 905,718 行 SFT，评测集从独立的 19,502 行 RLAIF 中确定性抽 2,000 条，精确 canonical overlap=0。四个模型只物化一次完全相同的 383,926 个 assistant tokens。

| 模型 | 参数 | Token-weighted loss | PPL |
|---|---:|---:|---:|
| Teacher 768 | 63.912M | 1.822442 | 6.186949 |
| Student init 512 | 30.025M | 2.057247 | 7.824396 |
| CE control 512 | 30.025M | 2.056352 | 7.817403 |
| KD 512 | 30.025M | 2.061657 | 7.858981 |

本次 KD 相对同预算 CE 的 loss **上升 0.258%**、PPL **上升 0.532%**，没有质量提升。教师明显优于学生说明存在可蒸馏信息，但当前 50k/1-epoch 小预算、`α=0.5,T=1.5` 组合可能让软目标正则过强，CE continuation 本身也只带来很小变化。正确后续是另设 validation split 搜索 `α/T` 和训练长度，再一次性报告 test；不能在这 2,000 条 holdout 上反复调参后仍称其为独立测试。面试口径是“完成白盒 logits KD 与公平负对照，当前配置收益不显著”，不是编造改善。

## 5. DPO 的独立 holdout 结果

DPO 数据按 prompt 分组后得到 15,456 个训练 prompt group 与 1,710 个 eval group，防止同一 prompt 的不同 chosen/rejected pair 跨集合。最终在固定前 1,000 个 eval pair 上比较共同 `full_sft` reference 和 DPO checkpoint：

| 模型 | Chosen preference accuracy | Mean implicit reward margin | Positive margin rate | DPO loss |
|---|---:|---:|---:|---:|
| Full SFT base | 45.70% | 0 | 0 | 0.693147 |
| DPO | 45.70% | -0.001024 | 51.00% | 0.694702 |

本次 DPO 没有提升 chosen preference accuracy，holdout DPO loss 相对 base 上升 0.224%。正 implicit margin 比例略高于随机，但均值为负，说明少量较大的负 margin 抵消了正样本。面试口径应是“完成 prompt-grouped DPO 与独立评测，当前小预算配置无显著收益”，不能把训练 loss 下降或 51% 正 margin rate 单独包装成整体提升。

## 6. MoE 分支能支撑的结论

MoE 分支为 4 experts、Top-1 routing。总参数 198.417M，估算每 token 激活参数 63.937M；因此它验证了“总容量扩展约 3.10 倍，每 token 激活规模与 Dense 接近”。最终 seed 42 固定批次审计的聚合专家负载为 `[21.09%, 24.02%, 29.59%, 25.29%]`，CV=12.23%，8 个 router tensor 与 96 个 expert tensor 的梯度均为 finite，未观察到该 batch 上的路由崩塌。此前手工随机批次得到 CV=6.22%；二者差异也说明单批路由数字不是总体负载估计。

1024-token prompt + 128-token greedy decode 的同机基准：

| 架构 | Cached tok/s | Uncached tok/s | Cached peak MiB | Uncached peak MiB |
|---|---:|---:|---:|---:|
| Dense | 130.87 | 145.44 | 448.46 | 453.59 |
| partial MoE | 88.06 | 62.93 | 1212.42 | 1217.45 |

所以这个教学级 MoE 实现并没有带来实测加速：权重显存约为 Dense 的 2.70 倍，朴素 Python/token dispatch 也使吞吐更低。它没有 expert parallel、all-to-all、capacity factor 与 fused dispatch。面试口径应是“实现并短程验证稀疏 MoE”，而不是“MoE 比 Dense 更快/更准”。

后训练统一放在 Dense，不是因为 LoRA/DPO/RL 与 MoE 原理不兼容，而是因为当前 MoE 只训练了单 epoch 的 10.52%，没有同质量 SFT 初始化；RL 还需同时驻留 current/reference policy、rollout 激活和 optimizer state，在单张 4090 上会放大 198M 总参数与朴素 dispatch 的成本。若把四算法分别跑在这个 partial MoE 上，结果会同时混入基座质量、router 漂移和实现吞吐三个混杂变量。Dense 主线因此用于隔离后训练算法变量，MoE 分支只回答容量、稀疏激活、路由负载和系统代价；有 expert parallel 与完整 MoE SFT 后，才值得增加 Dense-vs-MoE 的独立二维消融。

## 7. KV Cache、GQA 与 Flash/YaRN 实测

KV Cache 有两个精确一致性测试：增量 cached logits 与 full forward 对齐，cached/uncached greedy token 完全一致。在 8 Query heads / 4 KV heads 下，4096+32 tokens 的 BF16 KV Cache 理论值为 48.375 MiB，对应 MHA 为 96.75 MiB，理论降低 50%。

原语级 Flash 审计在 BF16 `[B=2,H=8,S=1024,D=96]` 上强制两种 SDPA backend，并通过 profiler 观察到 `aten::_scaled_dot_product_flash_attention` 与 CUDA `flash_fwd_kernel`：

| SDPA backend | Median ms | Peak MiB |
|---|---:|---:|
| Forced Flash | 0.2069 | 12.06 |
| Forced math | 1.3551 | 190.14 |

该 shape 下 Flash 约快 6.55 倍。它证明 RTX 4090/PyTorch 2.6 在这个 dtype/shape 选择了 Flash kernel，不等于本项目手写 FlashAttention-2，也不保证所有 shape 都命中。

4096-token prompt +32 decode 的功能基准：

| 位置方案 | Cached tok/s | Uncached tok/s | Cached peak MiB | Uncached peak MiB |
|---|---:|---:|---:|---:|
| RoPE | 130.97 | 106.81 | 571.04 | 542.54 |
| YaRN | 122.15 | 103.53 | 571.04 | 542.54 |

两种模式都能完成长序列前向和生成；YaRN 在该系统基准没有吞吐收益。这里只能说长位置外推机制可运行，不能据此声称长文本任务准确率提高。最终验收又对实际 `full_sft` 模型执行 BF16、1024-token forward，强制 Flash backend 后 forward 成功、logits finite，profiler 命中 `aten::_scaled_dot_product_flash_attention` 与 CUDA Flash kernel。

## 8. Agent/RLVR 数据与冷启动门禁

审计上游 `agent_rl.jsonl` 后发现：20,000 条可验证样本均为 calculator，另外 19,988 条既无 tools 也无 ground truth。后者不能被当作严格 Tool-Use RLVR 数据。因此新构建了与本地确定性工具环境一致的 896/224 训练/评测划分，覆盖 calculator、unit、weather、time、exchange、translation 和 multi-tool 七类，每类为 128/32，精确问题交集为 0。

全量重放审计又对 1,120 条轨迹逐条检查了工具可用性、参数合法性、确定性执行、oracle observation 等值、必需工具覆盖以及执行证据对 GT 的覆盖，最终 `total_failures=0`。该审计曾真实捕获 multi-tool JSON 被“只看最后一行”的假阴性，修复为“自然语言最终答案只验证 final region，结构化工具证据验证全量 JSON”后再归零。

通用 SFT 可能不会生成合法工具协议，从而使 G 个 rollout 全错，优势恒为 0。当前流程因此加入了训练集专用 Agent SFT 冷启动：896 条多工具 oracle 轨迹 + 512 条经表达式重算校验的数学轨迹，评测样本使用数为 0。四算法只能从这个共同 `agent_sft` 出发。

128 条固定 readiness 结果：

| 任务/初始化 | Task acc | Format | Tool valid | Tool execution | Required coverage | Evidence coverage | DAPO effective groups |
|---|---:|---:|---:|---:|---:|---:|---:|
| Math / full SFT | 0.00% | 81.25% | 71.09% | 69.53% | 48.44% | 10.81% | 0.00% |
| Math / Agent SFT | 59.38% | 92.97% | 100.00% | 97.27% | 100.00% | 85.68% | 15.63% |
| Basic Tool / full SFT | 7.03% | 84.38% | 51.17% | 49.61% | 41.80% | 20.70% | 15.63% |
| Basic Tool / Agent SFT | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 0.00% |

基础工具集在 Agent SFT 后饱和，因此不能用于比较 RL 算法。项目另建了 384/96 的六类组合 Tool-Use train/eval，`agent_sft_oracle_rows_used=0`、问题交集为 0，全量 480 条 oracle 重放 `total_failures=0`。训练集前 128 条校准中 task accuracy=0.781%、tool execution=65.63%、required coverage=48.18%、evidence coverage=30.86%、DAPO effective group rate=3.125%；这组数据难但没有完全失去学习信号。

多轮序列化审计还找到一个真实 tokenizer 非 round-trip token：原始 assistant action 为 36 tokens，canonical decode→encode 后变成 38。当前实现保留精确 sampled-token ledger，只拼接环境 observation suffix，因此 behavior/current log-prob 的 action ID 仍逐 token 对齐。

## 9. 当前可证明的工程正确性

远端单元测试为 31/31 通过，覆盖：

- KV Cache 增量与全量前向一致；
- RMSNorm 数值、YaRN 长位置频率、GQA KV 形状与 MoE router 梯度；
- KD 的同 logits 零 KL 和 student finite gradient；
- DPO 在 policy=reference 时为 `log 2`，增大 chosen 相对 margin 会降低目标；
- GRPO/CISPO/DAPO/GSPO 的 ratio、clip、token/sequence reduction、动态采样和 soft-overlong；
- 工具参数/执行/必需工具/执行结果证据覆盖，以及数字子串 reward hacking、“调错工具后猜对答案”、未闭合轨迹和超长左截断防护；
- 多轮 action mask 保留每一轮 assistant action/EOS，同时排除所有 tool observation；
- Agent SFT oracle 表达式与 ground truth 的重算一致性。

KD、DPO、冷启动前后评测、架构验证和两套 RLVR Formal 均已完成。训练组统计只用于解释机制与成本，所有泛化结论均取共同 fixed holdout。

## 10. 数学四算法 Formal 结果

共同初始化为 `agent_sft`，每个算法运行 3 个独立训练 seed，每个 run 最多接受 10 个更新组；每组 4 条轨迹、复用 2 个 policy epochs，学习率为 `1e-7`。训练聚合如下：

| 算法 | 训练组 Reward | 训练组准确率 | 候选组 | 动态接受率 | Wall time |
|---|---:|---:|---:|---:|---:|
| GRPO | -0.5167 ± 0.0577 | 24.17% ± 2.89% | 10.0 | — | 32.41 ± 0.62 s |
| CISPO | -0.5167 ± 0.0577 | 24.17% ± 2.89% | 10.0 | — | 32.79 ± 0.32 s |
| DAPO | 0.0667 ± 0.1893 | 53.33% ± 9.46% | 160.3 ± 43.7 | 8.23% ± 1.35% | 487.04 ± 137.22 s |
| GSPO | 0.0500 ± 0.0500 | 52.50% ± 2.50% | 10.0 | — | 28.77 ± 2.58 s |

DAPO 三个 seed 实际消耗 111、194、176 个候选组才分别保留 10 个有效组。它的训练 Reward/准确率是“只看混合成功组”的条件统计，不能与未筛选算法横向比较；可比较的主要结论是动态采样用约 16 倍候选组和约 15 倍 wall time 换取非零组内优势。

固定评测对共同 baseline 与 12 个 RL checkpoint 各运行 64 个 prompt × 3 个解码 seed，共 2,496 条轨迹。共同 baseline 与全部四算法的 Reward 都是 -0.03125，task/answer accuracy 都是 48.4375%，格式正确率 75.00%，工具执行率 95.3125%，响应长度 80.19 tokens；四算法对 baseline 的任务指标 delta 全为 0，跨训练 seed 标准差也为 0。固定集 k3 KL 仅为 `0–5.43e-9`。

这说明当前 `10 updates / 1e-7` 配置完成了机制与成本验证，但没有让概率变化大到足以改变固定随机种子的生成结果。训练批次上的 selected-group Reward 不代表泛化，本项目不据此宣称 DAPO/GSPO 优于基线。

## 11. 组合 Tool-Use 四算法训练结果

共同初始化、训练 seed 和更新预算与数学实验一致。组合 Tool-Use 的严格成功更稀疏，训练聚合如下：

| 算法 | 训练组 Reward | 训练组准确率 | 零方差组比例 | 候选组 | Wall time |
|---|---:|---:|---:|---:|---:|
| GRPO | -0.9333 ± 0.0764 | 3.33% ± 3.82% | 90.00% ± 10.00% | 10.0 | 28.97 ± 0.86 s |
| CISPO | -0.8667 ± 0.1155 | 6.67% ± 5.77% | 86.67% ± 11.55% | 10.0 | 28.80 ± 0.69 s |
| DAPO | -0.0333 ± 0.0289 | 48.33% ± 1.44% | 0% | 103.7 ± 17.6 | 285.19 ± 47.06 s |
| GSPO | -0.9667 ± 0.0289 | 1.67% ± 1.44% | 93.33% ± 5.77% | 10.0 | 28.16 ± 0.44 s |

DAPO 三个 seed 分别使用 118、109、84 个候选组，平均动态接受率 14.70% ± 3.86%。这证明动态采样能在大多数普通组全错、相对优势为零时找到可学习组，但约需 10.4 倍候选组和 9.9 倍 wall time。表中的 DAPO 高准确率是筛选条件造成的训练统计；泛化必须由下方同一固定 holdout 判断。

固定评测共 `13 checkpoints × 64 prompts × 3 decode seeds = 2,496` 条轨迹：

| 模型/算法 | Reward | Task/Answer acc | Format | Tool valid | Execution | Required | Evidence | Avg len | Unfinished |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Agent SFT baseline | -0.88542 | 5.7292% | 97.9167% | 99.4792% | 71.6146% | 56.6840% | 35.9375% | 92.34 | 1.0417% |
| GRPO | -0.88542 | 5.7292% | 97.9167% | 99.4792% | 71.6146% | 56.6840% | 35.9375% | 92.34 | 1.0417% |
| CISPO | -0.88542 | 5.7292% | 97.9167% | 99.4792% | 71.6146% | 56.6840% | 35.9375% | 92.34 | 1.0417% |
| DAPO | -0.88542 | 5.7292% | 97.9167% | 99.4792% | 71.6146% | 56.6840% | 35.9375% | 92.34 | 1.0417% |
| GSPO | -0.88542 | 5.7292% | 97.9167% | 99.4792% | 71.6146% | 56.6840% | 35.9375% | 92.34 | 1.0417% |

四算法所有任务指标相对 baseline 的 delta 都为 0，跨训练 seed 标准差也为 0；固定集 k3 KL 为 `0–1.53e-8`。因此当前小预算只证明 DAPO 可恢复非零学习信号并量化其额外成本，不证明准确率改善。下一轮应在独立 validation 上增加 accepted updates、调学习率/clip，并引入由易到难的组合任务课程。

## 12. 最终验收

- Agent SFT mask：128 个固定样本，9,407 个 supervised tokens，mismatch=0，zero-supervision=0；
- MoE：聚合负载 `[21.09%, 24.02%, 29.59%, 25.29%]`，CV=12.23%，8 个 router 与 96 个 expert gradient tensors 全部 finite；
- 多轮序列化：真实 token id 130 的 `decode→encode` 不可逆反例中，36 个 sampled action IDs 全部精确保留；
- 工具数据：基础 1,120 行 + challenge 480 行，全量 deterministic oracle replay failures=0；
- 实际模型 Flash：forced Flash forward 成功、logits finite、profiler 命中 Flash kernel；
- 回归测试：31/31 通过；机器证据索引 7 个 pipeline markers 全 true，`missing=[]`。

总入口为 `out/run_meta/experiment_evidence.json`，关键审计与日志均有 SHA-256。原始 fixed-set 逐轨迹记录保存在 `out/eval_rlvr/math/trajectories.jsonl` 与 `out/eval_rlvr/tool/trajectories.jsonl`。
