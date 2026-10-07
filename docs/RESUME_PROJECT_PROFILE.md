# 可验证工具调用数学 Agent 后训练系统：简历项目包装

> 项目仓库：[shelly-0o0/miniLLM-RL](https://github.com/shelly-0o0/miniLLM-RL/tree/gsm8k-agentic-rl)
> 推荐定位：大模型后训练 / Agentic RL / RLVR / LLM Training Infrastructure
> 表述原则：只使用已经完成并通过审计的实验，不把 Additional-SFT 的结果归因给 GRPO，也不把单次训练结果表述为跨种子普适结论。

## 一、简历可直接使用版本

### 面向可验证工具调用的数学 Agent 后训练与行为分析｜项目负责人

- **背景与目标：** 基于 MiniMind-64M 与 Qwen3-4B-Base 搭建端到端 Agent 后训练系统，围绕 GSM8K 研究行为冷启动、SFT 数据分布及稀疏 RLVR 奖励对工具调用和数学推理的影响；设计 Base、Agent-SFT、GRPO(A/B)、Additional-SFT 五臂对照，并保持官方 1,319 题 test 全程隔离。
- **数据与环境：** 从 7,473 条训练题构建题目级互斥 A/B 数据池和 7,378 条可验证 Agent oracle；实现 `calculate_math` 安全执行环境、真实多轮 tool-call/observation 回填和严格 verifier，联合校验答案、工具合法性、真实执行、必需工具覆盖与证据一致性，阻断伪造 observation、答案猜测和数字子串等 reward hack。
- **训练与工程：** 实现 assistant-only SFT、NF4 QLoRA/PEFT、`G=8` 在线 GRPO，以及 GRPO/CISPO/DAPO/GSPO 统一实验链；在 8×RTX 4090D 环境完成并行训练、断点恢复、原子分片合并和 fail-closed 审计，累计生成 64.6 万条 Stage 1 rollout、7,780 万 token、117.26 GPU-hours，完整回归测试 72/72 通过。
- **实验结果：** Qwen3-4B 上，Agent-SFT 后 strict-GRPO 将官方测试严格成功率由 **0.910% 提升至 2.654%（+1.744 pp，paired 95% CI [+0.758,+2.729]）**；同题高质量 oracle 的 Additional-SFT 达到 **42.532% strict / 99.040% 工具执行率**。进一步量化 GRPO(B) **87.76% 零方差组**，定位性能瓶颈为奖励密度而非 KL/数值不稳定，并形成 grounded shaped→strict、动态采样与自适应组大小的改进方案。

**技术栈：** Python、PyTorch、Transformers、PEFT、bitsandbytes、QLoRA/NF4、Hugging Face Datasets、CUDA、GSM8K、GRPO/RLVR、Git、Linux、tmux

## 二、版面紧张时的四行精简版

### 可验证工具调用数学 Agent：QLoRA SFT + GRPO｜项目负责人

- 基于 Qwen3-4B-Base 搭建 Agent-SFT、在线 `G=8` GRPO 与官方测试闭环；将 GSM8K 训练集确定性拆分为互斥 A/B 池，保留 1,319 道 test 仅用于最终评测。
- 实现安全 calculator、多轮工具调用环境与严格 RLVR verifier，联合验证答案、工具执行、覆盖及证据一致性；支持 assistant-only mask、NF4 QLoRA、冻结 reference、KL 门禁和可恢复训练。
- 在 8×RTX 4090D 上完成五臂对照：GRPO 将 strict accuracy 从 **0.910% 提升至 2.654%**；Additional-SFT 达到 **42.532% strict / 99.040% tool execution**，揭示可靠 oracle 与稀疏在线奖励的监督效率差异。
- 通过逐题 bootstrap/McNemar、数据与 checkpoint 哈希、原子分片合并和 72 项回归测试保证实验可审计；量化 **87.76% 零方差组**并定位 GRPO 的有效信号瓶颈。

## 三、偏算法岗位版本

### 可验证 Agentic RLVR：Group-Relative Policy Optimization 与奖励稀疏性研究

- 构建“策略生成工具动作—环境执行—observation 回填—继续决策—轨迹验证—策略更新”的真实 Agentic RL 闭环，而非仅对答案格式打分。
- 从同一 Agent-SFT 起点比较 GRPO、CISPO、DAPO、GSPO；MiniMind-64M 的 12 个正式 run 共生成 645,696 条轨迹，DAPO 在 official test 上达到 **3.319% strict**，较 Agent-SFT 提升 **1.120 pp**。
- 在 Qwen3-4B 上设计 A/B 互斥五臂实验；GRPO(A/B) 相对 Agent-SFT 均取得配对显著的小幅增益，但 A/B 差值仅 **0.076 pp、McNemar p=1.0**，避免把随机波动误写成泛化提升。
- 从组内优势公式出发定位 84%–88% 零方差组导致的梯度稀疏，结合工具执行漏斗、KL k3、rollout log-prob 对齐和训练前后行为分布提出 grounded reward curriculum 与动态采样方案。

## 四、偏训练系统岗位版本

### 消费级 GPU 上的 Qwen3-4B Agent 后训练与可审计实验平台

- 在单张 24GB RTX 4090D 上以 NF4 量化基座承载 policy/reference LoRA 视图，Qwen GRPO pilot 峰值约 14.2 GiB；将同一 prompt 的首轮 `G=8` 生成合并为 GPU batch，工具调用后的轨迹独立续写。
- 实现 PEFT adapter 初始化/恢复、冻结 reference、action-only log-prob、padding-safe action ledger、KL 安全门禁、checkpoint 元数据与原子结果合并，支持多 GPU 独立实验并行而不混写产物。
- 排查并修复 tied `lm_head` 未被 `all-linear` LoRA 覆盖、训练/重算侧 dropout 不一致、PAD 污染动作 token、SFT prefix 不一致及单步 warmup 学习率为零等问题。
- 建立数据边界、SHA-256 manifest、1,319 题完整覆盖、配置一致性及非有限 tensor 的 fail-closed 审计，最终 72/72 自动化测试通过。

## 五、30 秒面试介绍

这个项目不是单纯调用现成 Trainer 跑 GRPO，而是从数据、工具环境、轨迹验证、SFT 冷启动、在线 rollout 到统计评测搭建了一套可审计的数学 Agent 后训练系统。模型必须生成 calculator 调用，环境真实执行并返回 observation，最终奖励同时检查答案和工具证据。我在 64M 模型上做了四种 group-relative 算法、三个训练种子的比较，又在 Qwen3-4B 上完成 Base、Agent-SFT、GRPO A/B 和追加 SFT 五臂实验。结果显示 GRPO 有统计可测但绝对值较小的提升，而可靠 oracle 的追加 SFT 显著更有效；进一步分析发现约 88% 的 GRPO 组没有奖励方差，所以瓶颈是 on-policy 成功密度，而不是训练没更新或 KL 爆炸。

## 六、项目最值得追问的技术点

### 1. 为什么称为 Agentic RL，而不是普通数学 RL？

动作不只有最终答案。策略先生成结构化工具调用，外部环境解析并执行表达式，把执行结果作为新 observation 回填，策略再继续决策。奖励由完整交互轨迹决定，因此存在动作、状态变化、环境反馈和轨迹级信用分配。

### 2. 为什么 SFT 不执行工具，但仍能学习 Agent 行为？

SFT 是对离线可靠轨迹进行行为克隆：assistant 的工具调用和最终回答进入 loss，环境产生的 tool observation 只作为条件上下文。在线 RL 和评测阶段才真正调用环境。这样既不会监督模型伪造工具结果，又能学习读取 observation 后继续回答。

### 3. 为什么 Additional-SFT(B) 远好于 GRPO(B)？

Additional-SFT 的每个 assistant token 都有正确 action oracle；strict-GRPO 只有完整轨迹全部正确时才有正奖励，而且 87.76% 的组内八条轨迹奖励完全相同，优势归一化后近似没有梯度。因此差异反映监督密度和探索可达性，而不能概括成“SFT 永远优于 RL”。

### 4. 为什么不能说 GRPO(B) 优于 GRPO(A)？

两者 strict accuracy 为 2.654% 和 2.578%，只差 0.076 个百分点；逐题配对区间跨 0，McNemar `p=1.0`，discordant success 仅 31:30。该差异与统计噪声一致。

### 5. 下一步怎样让 GRPO 更有效？

先把奖励改成由真实工具执行和 evidence 门控的 grounded shaping，再逐步退火到 strict reward；对全失败组自适应增加采样数，对混合成功组更新，对全成功组提升难度。同时以有效组率、工具执行、证据覆盖和 strict accuracy 设置训练放行门槛，避免 shaped reward 上升但 Agent 行为不改善。

## 七、简历表述红线

不要写：

- “GRPO 将 Qwen3-4B 提升到 42.53%”；42.53% 来自 Additional-SFT(B)。
- “GRPO(B) 显著优于 GRPO(A)”；两者差异没有统计证据。
- “使用了 veRL、vLLM、FSDP2 或 TRL”；当前正式实现没有依赖这些框架。
- “实现多卡线性加速”；实际采用的是多实验单卡并行，没有做 scaling-efficiency 测试。
- “达到 SOTA”；实验没有与公开 SOTA 在同等模型、提示和评测协议下比较。
- “Qwen3-4B 全参数训练”；正式路线为 NF4 QLoRA/PEFT。

更准确的表述是：

- “搭建并验证端到端 Agentic RLVR 实验闭环”；
- “在当前初始化与奖励条件下量化 GRPO 的增益和信号稀疏瓶颈”；
- “通过真实工具执行与严格 verifier 防止 answer-only 指标高估”；
- “形成可复现、可恢复、可审计的消费级 GPU 后训练流水线”。

## 八、推荐标题备选

按目标岗位选择一个，不要同时堆叠多个标题：

1. **面向可验证工具调用的数学 Agent 后训练与行为分析**（最均衡）
2. **Qwen3-4B Agentic RLVR：QLoRA SFT、GRPO 与奖励稀疏性研究**（偏算法）
3. **消费级 GPU 上的可审计 LLM Agent 后训练平台**（偏系统）
4. **Group-Relative RL for Tool-Using Math Agents**（英文简历）
