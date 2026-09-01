"""
训练工具函数集合

这个文件可以理解为训练脚本的公共底座，主要负责：
- 分布式训练初始化
- 随机种子固定
- 学习率调度
- checkpoint 保存与恢复
- 基础模型与 tokenizer 加载
- 断点续训时跳过已训练 batch
- reward model 的统一封装
"""

import os
import sys
import re

# 让 trainer 目录下的脚本可以直接运行，同时还能正确 import 上层模块
__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import random
import math
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import Sampler
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
from model.model_minimind import MiniMindForCausalLM


def get_model_params(model, config):
    """
    打印模型参数量信息。

    这个函数常用于初始化模型时做日志统计，帮助判断：
    - 总参数量有多少
    - 如果是 MoE 模型，单个 token 实际激活多少参数
    """
    # 模型总参数量，单位是 M
    total = sum(p.numel() for p in model.parameters()) / 1e6

    # 兼容不同 config 命名：
    # - 有些配置叫 n_routed_experts
    # - 有些配置叫 num_experts
    n_routed = getattr(config, 'n_routed_experts', getattr(config, 'num_experts', 0))

    # 每个 token 实际激活多少个专家
    n_active = getattr(config, 'num_experts_per_tok', 0)

    # 共享专家数
    n_shared = getattr(config, 'n_shared_experts', 0)

    # 统计一个 routed expert 的参数量
    # 通过名字里是否包含 'mlp.experts.0.' 来估算单个 expert 规模
    expert = sum(p.numel() for n, p in model.named_parameters() if 'mlp.experts.0.' in n) / 1e6

    # 统计一个 shared expert 的参数量
    shared_expert = sum(p.numel() for n, p in model.named_parameters() if 'mlp.shared_experts.0.' in n) / 1e6

    # base 表示除专家层之外的参数量
    base = total - (expert * n_routed) - (shared_expert * n_shared)

    # active 表示每个 token 实际参与计算的参数规模
    # MoE 模型里，虽然总参数很多，但每次只激活一部分专家
    active = base + (expert * n_active) + (shared_expert * n_shared)

    # 如果 active 小于 total，说明是稀疏激活的 MoE 模型
    if active < total:
        Logger(f'Model Params: {total:.2f}M-A{active:.2f}M')
    else:
        Logger(f'Model Params: {total:.2f}M')


def is_main_process():
    """
    判断当前进程是否是主进程。

    在分布式训练中只有 rank 0 进程负责：
    - 打印日志
    - 保存 checkpoint
    - 记录 wandb/swanlab
    """
    return not dist.is_initialized() or dist.get_rank() == 0


def Logger(content):
    """
    简单日志输出包装器。

    只在主进程打印，避免 DDP 多进程下日志重复刷屏。
    """
    if is_main_process():
        print(content)


def get_lr(current_step, total_steps, lr):
    """
    学习率调度函数。

    这里是一个 cosine decay 风格的调度：
    - 起点接近 lr
    - 终点降到 0.1 * lr
    """
    return lr * (0.1 + 0.45 * (1 + math.cos(math.pi * current_step / total_steps)))


def init_distributed_mode():
    """
    初始化分布式训练环境。

    逻辑：
    - 如果环境变量 RANK 不存在，说明当前不是 DDP 模式
    - 如果存在，则初始化 NCCL 进程组，并绑定当前 local_rank 对应的 GPU
    """
    if int(os.environ.get("RANK", -1)) == -1:
        return 0  # 非DDP模式

    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank


def setup_seed(seed: int):
    """
    固定随机种子，保证训练尽可能可复现。

    这里同时设置了：
    - random
    - numpy
    - torch
    - cudnn 的 deterministic/b benchmark 开关
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # deterministic=True 更可复现，但可能牺牲一点速度
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def lm_checkpoint(
    lm_config,
    weight='full_sft',
    model=None,
    optimizer=None,
    epoch=0,
    step=0,
    wandb=None,
    save_dir='../checkpoints',
    **kwargs
):
    """
    保存或加载训练 checkpoint。

    两种模式：
    1. model 不为 None：保存模型和训练状态
    2. model 为 None：尝试从 resume 文件恢复训练状态

    这里区分了两个文件：
    - ckp_path：只保存模型权重，适合推理或 warm start
    - resume_path：保存完整训练恢复信息，适合断点续训
    """
    os.makedirs(save_dir, exist_ok=True)

    # 根据是否使用 MoE 拼接文件名
    moe_path = '_moe' if lm_config.use_moe else ''
    ckp_path = f'{save_dir}/{weight}_{lm_config.hidden_size}{moe_path}.pth'
    resume_path = f'{save_dir}/{weight}_{lm_config.hidden_size}{moe_path}_resume.pth'

    if model is not None:
        # 保存模式
        # DDP 下真实模型在 model.module 里
        raw_model = model.module if isinstance(model, DistributedDataParallel) else model

        # torch.compile 后模型可能包了一层 _orig_mod
        raw_model = getattr(raw_model, '_orig_mod', raw_model)

        # 取出 state_dict，并转成半精度 + CPU，减小文件体积
        state_dict = raw_model.state_dict()
        state_dict = {k: v.half().cpu() for k, v in state_dict.items()}

        # 先写临时文件，再原子替换，避免保存中断导致文件损坏
        ckp_tmp = ckp_path + '.tmp'
        torch.save(state_dict, ckp_tmp)
        os.replace(ckp_tmp, ckp_path)

        # 记录 wandb / swanlab 的 run id，方便恢复实验
        wandb_id = None
        if wandb:
            if hasattr(wandb, 'get_run'):
                run = wandb.get_run()
                wandb_id = getattr(run, 'id', None) if run else None
            else:
                wandb_id = getattr(wandb, 'id', None)

        # resume_data 用于完整恢复训练状态
        resume_data = {
            'model': state_dict,
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'step': step,
            'world_size': dist.get_world_size() if dist.is_initialized() else 1,
            'wandb_id': wandb_id
        }

        # 额外状态也可以一起保存，比如 scheduler、scaler 等
        for key, value in kwargs.items():
            if value is not None:
                if hasattr(value, 'state_dict'):
                    raw_value = value.module if isinstance(value, DistributedDataParallel) else value
                    raw_value = getattr(raw_value, '_orig_mod', raw_value)
                    resume_data[key] = raw_value.state_dict()
                else:
                    resume_data[key] = value

        # 同样采用临时文件 + 原子替换
        resume_tmp = resume_path + '.tmp'
        torch.save(resume_data, resume_tmp)
        os.replace(resume_tmp, resume_path)

        # 清理内存，防止长训练时显存/内存堆积
        del state_dict, resume_data
        torch.cuda.empty_cache()

    else:
        # 加载模式：尝试读取 resume 文件
        if os.path.exists(resume_path):
            ckp_data = torch.load(resume_path, map_location='cpu')

            # 如果保存时和当前的 GPU 数量不一致，按 world_size 粗略换算 step
            saved_ws = ckp_data.get('world_size', 1)
            current_ws = dist.get_world_size() if dist.is_initialized() else 1
            if saved_ws != current_ws:
                ckp_data['step'] = ckp_data['step'] * saved_ws // current_ws
                Logger(f'GPU数量变化({saved_ws}→{current_ws})，step已自动转换为{ckp_data["step"]}')

            return ckp_data

        return None


def init_model(
    lm_config,
    from_weight='pretrain',
    tokenizer_path='../model',
    save_dir='../out',
    device='cuda'
):
    """
    初始化基础模型和 tokenizer。

    作用：
    - 加载 tokenizer
    - 构建 MiniMindForCausalLM
    - 可选加载已有权重
    - 打印参数量统计
    - 把模型移动到指定设备
    """
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    model = MiniMindForCausalLM(lm_config)

    # from_weight != 'none' 时表示需要从已有权重继续训练或做微调
    if from_weight != 'none':
        moe_suffix = '_moe' if lm_config.use_moe else ''
        weight_path = f'{save_dir}/{from_weight}_{lm_config.hidden_size}{moe_suffix}.pth'
        if not os.path.exists(weight_path):
            # Keep immutable group checkpoints loadable across both naming
            # conventions used by the agent experiments:
            #   weight_768_groups50.pth and weight_groups50_768.pth.
            group_match = re.match(r'^(.*)_groups(\d+)$', str(from_weight))
            if group_match:
                reordered = (
                    f'{save_dir}/{group_match.group(1)}_{lm_config.hidden_size}'
                    f'{moe_suffix}_groups{group_match.group(2)}.pth'
                )
                if os.path.exists(reordered):
                    weight_path = reordered

        # map_location=device 保证在当前设备上加载
        weights = torch.load(weight_path, map_location=device)

        # strict=False 允许部分 key 不匹配，适合预训练 -> 微调、普通模型 -> MoE 等场景
        model.load_state_dict(weights, strict=False)

    get_model_params(model, lm_config)
    Logger(f'Trainable Params: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.3f}M')
    return model.to(device), tokenizer


class SkipBatchSampler(Sampler):
    """
    自定义 Batch Sampler。

    它的作用是：
    - 包装一个基础 sampler
    - 按 batch_size 组装成 batch
    - 跳过前 skip_batches 个 batch

    典型用途：
    - 断点续训时跳过已经训练过的 batch
    - 避免从 epoch 中间恢复时重复计算前面步骤
    """
    def __init__(self, sampler, batch_size, skip_batches=0):
        self.sampler = sampler
        self.batch_size = batch_size
        self.skip_batches = skip_batches

    def __iter__(self):
        batch = []
        skipped = 0

        for idx in self.sampler:
            batch.append(idx)

            # 收集够一个 batch 之后再决定是跳过还是输出
            if len(batch) == self.batch_size:
                if skipped < self.skip_batches:
                    skipped += 1
                    batch = []
                    continue

                yield batch
                batch = []

        # 处理最后一个不足 batch_size 的尾巴 batch
        if len(batch) > 0 and skipped >= self.skip_batches:
            yield batch

    def __len__(self):
        # 计算总 batch 数，使用向上取整
        total_batches = (len(self.sampler) + self.batch_size - 1) // self.batch_size
        return max(0, total_batches - self.skip_batches)


class LMForRewardModel:
    """
    奖励模型封装器。

    这个类的作用不是训练主模型，而是在强化学习阶段提供 reward score。
    典型使用场景：
    - Agent RL
    - PPO / GRPO / 其他需要 reward model 打分的流程

    它把底层 reward model 封装成统一接口：
    - 输入：messages + response
    - 输出：一个标量分数
    """
    def __init__(self, model_path, device="cuda", dtype=torch.float16):
        # reward model 也可能是自定义实现，因此这里启用 trust_remote_code
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_path, torch_dtype=dtype, trust_remote_code=True)

        # reward model 一般只做推理，不参与梯度更新
        self.model = self.model.to(device).eval()
        self.device = device

    @torch.no_grad()
    def get_score(self, messages, response):
        """
        对一段候选回复打分。

        参数：
        - messages：对话历史，通常是一个 message list
        - response：当前候选回答

        处理逻辑：
        1. 把多轮历史整理成一个可供 reward model 读取的文本上下文
        2. 构造两轮形式的评估消息：user + assistant
        3. 调用 reward model 的 get_score 方法
        4. 将分数裁剪到 [-3, 3]，避免 reward 过大导致训练不稳定
        """
        # 将历史对话压缩成文本
        # 这里忽略最后一条 message，因为最后一条通常就是当前问题本身
        history_text = "\n".join([f"{m['role']}: {m['content']}" for m in messages[:-1]])

        # 最后一条消息作为当前新问题
        last_query = messages[-1]['content'] if messages else ""

        # 将历史上下文和当前问题拼成一个统一的用户输入
        message_context = f"{history_text}\n以上是对话历史。我的新问题是：\n{last_query}" if history_text else last_query

        # reward model 一般期望一个简洁的 user/assistant 二轮结构
        eval_messages = [
            {"role": "user", "content": message_context},
            {"role": "assistant", "content": response}
        ]

        # 由 reward model 计算分数
        score = self.model.get_score(self.tokenizer, eval_messages)

        # 对输出分数做裁剪，防止 reward scale 太极端
        return max(min(score, 3.0), -3.0)
