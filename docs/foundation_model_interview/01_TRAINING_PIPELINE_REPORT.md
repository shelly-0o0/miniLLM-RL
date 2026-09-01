# 工作报告一：Pretrain、SFT、LoRA 与知识蒸馏

## 1. 项目范围和个人工作边界

官方 MiniMind 已提供 Pretrain、全参 SFT、手写 LoRA、白盒蒸馏、AMP、梯度累积、DDP 和 checkpoint。本项目在此基础上完成代码审计、端到端复现方案、Assistant-only mask 验证、有效 batch 对齐、训练阶段衔接与量化协议。面试中应说“基于官方仓库复现并验证训练链路”，不能把上游脚本描述为从零原创。

## 2. Pretrain

### 2.1 目标

对 token 序列 `x₁…x_T` 最大化自回归似然：

```text
L_pretrain = - Σ_t log pθ(x_t | x_<t)
```

代码中输入含 BOS/EOS，pad label 设为 `-100`。模型 forward 使用 `logits[..., :-1, :]` 预测 `labels[..., 1:]`，因此第 `t` 个 hidden state 预测下一个 token。

### 2.2 为什么必须先预训练

SFT 主要学习行为和条件分布，无法低成本替代大规模语言建模。随机模型直接 SFT 容易只记模板、事实覆盖差、生成分布脆弱。对后续 RL 而言，基础策略若在题目上全错，组内 reward 方差为零，GRPO 家族没有学习信号。

### 2.3 关键工程点

- 数据质量比单纯 token 数更重要：去重、语言/领域配比、污染检查、长短样本分桶。
- 有效 token 数而非样本数决定主要训练预算。
- 学习率与全局 batch 绑定；改变 GPU 数后不能只保持 per-GPU batch。
- 记录 tokens/s、MFU、峰值显存、loss、grad norm、学习率、数据消费量。
- 断点恢复必须包含 optimizer、scaler、scheduler/step、RNG 和数据位置；MiniMind 已保存主要训练状态，但严格复现还应持久化 sampler/RNG。

## 3. SFT 与 Assistant-only Loss Mask

### 3.1 目标

对对话上下文 `c` 和 assistant 回复 `y`：

```text
L_SFT = - Σ_t m_t log pθ(y_t | c, y_<t) / Σ_t m_t
```

`m_t=1` 只在 assistant token，system/user/tool/pad 为 0。MiniMind 用 label=`-100` 表示 `m_t=0`。

### 3.2 为什么不能把 prompt 一起算 loss

模型的任务是根据 prompt 生成回答，不是复述 prompt。若 user token 参与 loss：

- 大量梯度被输入模板占据，浪费容量；
- 模型可能学会续写/模仿用户而非回答；
- 固定 system prompt 被过度记忆，泛化变差；
- 工具 observation 是环境给出的，不应被当作策略 action 优化。

### 3.3 最危险的实现错误

Mask 与 causal shift 错一位。正确检查不是只看 `labels != -100` 的数量，而是逐 token 打印：当前位置输入 token、下一 token、shift 后 label、mask。还要覆盖多轮 assistant、空回复、截断在 assistant 边界、tool call 和中文多 token 标签。

当前 `SFTDataset` 通过 tokenizer 对 `BOS + assistant` 与 `EOS` 的 token 序列定位。若修改 chat template，必须重跑 mask 单测；字符串看似相同不代表 token 边界相同。

本项目用 `scripts/audit_sft_mask.py` 对实际 tokenizer 和数据逐样本重建 assistant 区间，检查监督位置、label/token 一致性、pad 泄漏和零监督样本，并保存可读 span 供人工抽查。它能发现边界实现错误，但不能替代对 chat template 语义和截断样本的人工审计。

## 4. LoRA

### 4.1 原理

冻结原权重 `W₀∈R^{d_out×d_in}`，只学习低秩增量：

```text
W = W₀ + ΔW
ΔW = (α/r) B A
A ∈ R^{r×d_in}, B ∈ R^{d_out×r}, r << min(d_in,d_out)
```

参数量从 `d_out×d_in` 降到 `r(d_in+d_out)`。初始化常令一侧为零，使训练开始时 `ΔW=0`，不破坏基模输出。

### 4.2 为什么使用

- 多领域适配只存小 adapter；
- optimizer state 和梯度显存显著降低；
- 适合快速验证数据与任务；
- 合并回权重后无额外线性层推理延迟。

### 4.3 不能夸大的地方

LoRA 降低的是可训练参数、梯度和 optimizer state，不自动消除基模权重与前向激活；长序列显存仍可能被 activation 主导。低 rank 也可能限制大幅能力迁移。应同时对比 trainable params、峰值显存、step time、目标集和通用集。

MiniMind 的 LoRA 是自定义 monkey patch，并非 PEFT；`torch.compile` 被主动关闭。面试官问生产化时，应提到 target module 白名单、adapter 合并、量化兼容、FSDP state dict 和多租户热加载。

## 5. 知识蒸馏

### 5.1 白盒蒸馏目标

教师与学生在温度 `T` 下：

```text
p_T = softmax(z_teacher / T)
q_T = softmax(z_student / T)
L_KD = T² KL(p_T || q_T)
L = α L_CE + (1-α) L_KD
```

温度升高会软化分布，暴露“非正确 token 之间的相对偏好”，即暗知识。`T²` 用于补偿 softmax 温度导致的梯度缩小。

### 5.2 为什么保留 CE

纯 KL 会继承教师错误和校准偏差；硬标签提供任务锚点。`α` 过高退化为普通 SFT，过低则学生过度模仿教师。应在独立验证集调 `α/T`，不能只看 distillation loss。

### 5.3 当前代码约束

- Teacher/student 共用 tokenizer；不同 tokenizer 时直接对 vocab 位置做 KL 没有语义。
- 教师在 `eval + no_grad`，但仍需额外显存和计算。
- KD 只在 assistant mask 上计算。
- Teacher vocab 被截到 student vocab 只是形状兼容，不解决词表映射问题。
- 小 student 的初始化 checkpoint 名由 hidden size 决定，运行前必须存在。

### 5.4 本机受控结果

512 学生相对 768 教师减少 53.02% 参数。在 50k 相同训练预算和 2,000 条独立 RLAIF holdout 上，CE loss/PPL 为 2.056352/7.817403，`α=0.5,T=1.5` 的 KD 为 2.061657/7.858981，即 KD 分别差 0.258%/0.532%。这说明“教师更强”和“当前 KD 超参有效”是两个不同命题：教师 loss 1.822442，确实更强；但学生在小预算下同时追硬标签和软分布没有自动获得收益。下一步应用独立 validation 调 `α/T/epochs`，不能把当前负结果隐藏，也不能在 test 上调参。

### 5.5 更大规模的改进方向

- Top-k logit distillation：只传 teacher 高概率 token，降低存储/通信。
- Offline logits：教师预计算，省在线显存但占磁盘并固定数据增强。
- Hidden/attention distillation：需层映射和投影，不一定优于 logit KD。
- Sequence-level/black-box KD：用强教师生成高质量轨迹，再做 SFT；无法访问 logits 时可用。
- On-policy distillation：让学生分布上的上下文也被教师纠正，减轻 exposure bias，但成本更高。

## 6. 混合精度

### 6.1 BF16 与 FP16

BF16 指数位与 FP32 相同，动态范围大，通常不需要 GradScaler；尾数短、精度低。FP16 尾数更多但指数范围小，梯度容易下溢，因此通常配合动态 loss scaling。

MiniMind 的 Pretrain/SFT/LoRA/KD 使用 autocast；FP16 时启用 GradScaler，裁剪前先 `unscale_`。BF16 是 Ampere 及以后 GPU 的首选。

### 6.2 必须保留 FP32 的位置

常见包括 optimizer state、loss 归约、softmax/log-softmax、归一化统计和敏感的 LM head。新增 RL 代码把 log-softmax 输入转为 FP32，减少 train/inference logprob 漂移。混合精度不能只看“没有 NaN”，还需对照关键 logits、loss 与梯度。

## 7. 梯度累积

把 `K` 个 micro batch 的 loss 除以 `K` 后依次 backward，再 optimizer step。近似全局 batch：

```text
B_global = B_per_gpu × K × world_size
```

“近似”是因为：

- dropout mask 不同；
- BatchNorm 统计不同（本模型无 BatchNorm）；
- token-mean loss 若每个 micro batch 有效 token 数差异大，简单除 K 不等价于全批 token mean；
- 梯度裁剪必须在累计完、且 FP16 unscale 后执行；
- Adam 的更新频率改变，学习率 schedule 必须按 optimizer step 而非 micro step。

对于变长 SFT/RL，更严格做法是累计 loss sum 和 valid-token count，最后按跨 micro batch 的总 token 数归一化。

## 8. DDP

每个 rank 持有完整模型与 optimizer；`DistributedSampler` 切分数据；参数梯度在 backward hook 中 bucket 化并 all-reduce，默认取平均。DDP 不自动切 input，也不降低单卡模型内存。

面试关键点：

- 一卡一进程通常优于 `DataParallel`；
- 所有 rank 必须走相同 collective 序列，否则 hang；
- 累积时可用 `no_sync()` 避免前 K-1 次无谓 all-reduce，且 forward 也需放入 context；MiniMind 当前未做此优化；
- 通信可与反向计算重叠，bucket 大小影响性能；
- 模型放不进单卡时应使用 FSDP/ZeRO、TP/PP，而不是继续加 DDP 卡数。

## 9. 各阶段如何衔接

```text
随机初始化
  → Pretrain：语言/知识分布
  → SFT：指令、角色、格式、工具冷启动
  → 可选 LoRA：低成本领域适配
  → 可选 KD：压缩或迁移 teacher 分布
  → DPO/RLVR：偏好、推理与可验证任务优化
```

LoRA 和 KD 不是必须串行。LoRA 常从 SFT 分叉；KD 可以用大 teacher 生成/提供分布来训练小 student；online RL 应从有基本任务成功率的冷启动模型开始。

## 10. 结果改善应如何报告

本机已完成的数字与仍缺少的对照必须分开：

| 阶段 | 本机已有证据 | 当前能说什么 | 仍不能说什么 |
|---|---|---|---|
| Pretrain | 64M Dense、2 epochs，最后记录训练 loss 1.8342 | 从随机权重完成优化并产出可用 SFT 初始化 | 没有独立 validation PPL，不能说泛化达到某水平 |
| SFT | 2 epochs；128 样本 mask mismatch=0；最后记录训练 loss 1.4291 | Assistant-only 目标与训练链路正确运行 | 训练 loss 不能代替通用指令评测 |
| LoRA | 0.393M / 0.61% 可训练参数；医疗 holdout loss/PPL 改善 7.62%/12.15% | 低训练参数预算改善该固定领域 holdout | 没有医疗安全和通用遗忘评测，不能外推 |
| KD | 学生参数少 53.02%；同预算 KD 比 CE loss/PPL 差 0.258%/0.532% | 白盒蒸馏与公平对照已完成，当前超参是负结果 | 不能把 teacher 更强等同为当前 KD 配置有效 |
| AMP | 全主线实际使用 RTX 4090 BF16，关键 log-softmax/KL 保持 FP32 | 混合精度路径稳定完成训练 | 没有 FP32 同预算吞吐/质量 A/B，不能报加速倍数 |
| DDP | 代码沿用上游 DDP/DistributedSampler 路径；本机仅 1 张 GPU | 能解释实现、有效 batch 和 all-reduce 语义 | 没有真实多卡 scaling efficiency，简历不能写“实测线性加速” |

若资源允许，Pretrain/SFT 还应补 validation PPL 与任务集；AMP 补 FP32/BF16 A/B；DDP 补相同全局 batch 的 1/2/4 卡 throughput。指标未跑就写成“后续实验”，不要以架构推导值冒充实测改善。

Scaling efficiency：

```text
E_N = throughput_N / (N × throughput_1)
```

不能只报“4 卡快 4 倍”；要报告 GPU 型号、序列长度、batch、precision、通信互联和是否包含 data loading/checkpoint。

## 11. 高频项目拷问

### 为什么 SFT 只 mask prompt，不 mask assistant 的 EOS？

EOS 是停止行为的一部分，应监督；否则模型可能难以学会正常结束。若样本因截断没有完整 EOS，需要单独统计截断率。

### 为什么 BF16 通常不用 GradScaler？

BF16 指数动态范围与 FP32 相近，梯度下溢风险小；GradScaler 主要解决 FP16 的窄指数范围。并非绝对不能用，而是收益通常有限。

### 梯度累积是否完全等于大 batch？

不完全。除了 dropout 和更新频率，变长序列下的归约权重尤其容易不等价。要说明 loss denominator。

### DDP 为何没有降低单卡显存？

因为每 rank 都复制完整参数、梯度和 optimizer state。要分片需 FSDP/ZeRO。

### KD 为什么乘 `T²`？

softmax 对 logits 的导数包含约 `1/T`，KL 梯度再引入尺度，`T²` 用来维持与硬标签损失相近的梯度量级。

### LoRA rank 越大越好吗？

表达能力增强但参数、显存、过拟合风险上升；rank 应与 target modules、数据规模和任务变化幅度联合调。只对少数层做高 rank 和对全层做低 rank 是不同预算分配。

### 如何证明 Assistant-only mask 正确？

展示 token 级可视化、边界单测、有效 token 比例，并验证 user/system/tool observation 的 label 全为 `-100`。仅展示 loss 曲线不够。
