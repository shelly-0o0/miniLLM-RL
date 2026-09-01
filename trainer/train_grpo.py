import os
import sys

__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import datasets  # noqa: F401  # Windows pyarrow/torch DLL conflict workaround (issue #771)
import argparse
import math
import re
import gc
import warnings
import torch
import torch.nn.functional as F
import torch.distributed as dist
from transformers import AutoTokenizer
from contextlib import nullcontext
from torch import optim
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import AutoModel
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from dataset.lm_dataset import RLAIFDataset
from trainer.trainer_utils import Logger, is_main_process, lm_checkpoint, init_distributed_mode, setup_seed, SkipBatchSampler, init_model, LMForRewardModel
from trainer.rollout_engine import create_rollout_engine
from trainer.policy_optimization import (
    compute_policy_loss,
    distributed_token_mean_scale,
    group_relative_advantages,
)
from trainer.experiment_logging import JsonlMetricLogger

warnings.filterwarnings('ignore')


def rep_penalty(text, n=3, cap=0.5):
    toks = re.findall(r"\w+|[^\w\s]", text.lower())
    grams = [tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)]
    return min(cap, (len(grams) - len(set(grams))) * cap * 2 / len(grams)) if grams else 0.0


def calculate_rewards(prompts, responses, reward_model):
    rewards = torch.zeros(len(responses), device=args.device)

    with torch.no_grad():
        reward_model_scores = []
        batch_size = len(prompts)

        for i in range(batch_size):
            for j in range(args.num_generations):
                response_idx = i * args.num_generations + j
                response = responses[response_idx]
                prompt = prompts[i]

                pattern = r"<\|im_start\|>(system|user|assistant)\s+(.*?)<\|im_end\|>"
                matches = re.findall(pattern, prompt, re.DOTALL)
                messages = [{"role": role, "content": content.strip()} for role, content in matches]
                answer = response
                rewards[response_idx] += 0.5 if 20 <= len(response.strip()) <= 800 else -0.5
                if '</think>' in response:
                    thinking_content, answer_content = response.split('</think>', 1)
                    rewards[response_idx] += 1.0 if 20 <= len(thinking_content.strip()) <= 300 else -0.5
                    rewards[response_idx] += 0.25 if response.count('</think>') == 1 else -0.25
                    answer = answer_content.strip()
                rewards[response_idx] -= rep_penalty(answer)

                score = reward_model.get_score(messages, answer)
                reward_model_scores.append(score)

        reward_model_scores = torch.tensor(reward_model_scores, device=args.device)
        rewards += reward_model_scores

    return rewards


def grpo_train_epoch(epoch, loader, iters, rollout_engine, ref_model, reward_model, start_step=0, wandb=None, use_sglang=False):
    last_micro_step = 0
    for step, batch in enumerate(loader, start=start_step + 1):
        prompts = batch['prompt']  # list[str], length B
        prompt_inputs = tokenizer(prompts, return_tensors="pt", padding=True, return_token_type_ids=False,
                                  padding_side="left", add_special_tokens=False).to(args.device)
        if args.max_seq_len:
            prompt_inputs["input_ids"] = prompt_inputs["input_ids"][:, -args.max_seq_len:]
            prompt_inputs["attention_mask"] = prompt_inputs["attention_mask"][:, -args.max_seq_len:]

        rollout_result = rollout_engine.rollout(
            prompt_ids=prompt_inputs["input_ids"],
            attention_mask=prompt_inputs["attention_mask"],
            num_generations=args.num_generations,
            max_new_tokens=args.max_gen_len,
            temperature=0.8,
        )
        outputs = rollout_result.output_ids
        completion_ids = rollout_result.completion_ids
        completions = rollout_result.completions
        old_per_token_logps = rollout_result.per_token_logps.to(args.device).detach()
        prompt_lens = rollout_result.prompt_lens.to(args.device)
        full_mask = (outputs != tokenizer.pad_token_id).long()
        logp_pos = prompt_lens.unsqueeze(1) - 1 + torch.arange(completion_ids.size(1), device=args.device).unsqueeze(0)

        rewards = calculate_rewards(prompts, completions, reward_model).to(args.device)  # [B*num_gen]

        # Match the policy/rollout autocast precision; otherwise numerical
        # differences on low-probability tokens masquerade as policy KL.
        with torch.no_grad(), autocast_ctx:
            ref_logits = ref_model(outputs, attention_mask=full_mask).logits[:, :-1, :]
            ref_per_token_logps = F.log_softmax(ref_logits.float(), dim=-1).gather(
                2, outputs[:, 1:].unsqueeze(-1)
            ).squeeze(-1).gather(1, logp_pos)

        if args.debug_mode and is_main_process() and step % args.debug_interval == 0:
            for i in range(len(prompts)):
                Logger(f"[DEBUG] step={step}, sample[{i}]")
                Logger('-'*100)
                Logger(f"{'=' * 30} [DEBUG] sample[{i}] CONTEXT_BEGIN {'=' * 30}")
                Logger(prompts[i])
                Logger(f"{'=' * 31} [DEBUG] sample[{i}] CONTEXT_END {'=' * 31}")
                for j in range(args.num_generations):
                    idx = i * args.num_generations + j
                    Logger(f"{'=' * 28} [DEBUG] gen[{j}] RESPONSE_BEGIN {'=' * 28}")
                    Logger(completions[idx])
                    Logger(f"{'=' * 29} [DEBUG] gen[{j}] RESPONSE_END {'=' * 29}")
                    Logger(f"[DEBUG] gen[{j}] reward={rewards[idx].item():.4f}")
                Logger('='*100)

        grouped_rewards = rewards.view(-1, args.num_generations)  # [B, num_gen]
        advantages = group_relative_advantages(rewards, args.num_generations)

        completion_pad_mask = rollout_result.completion_mask.to(args.device).bool()
        is_eos = (completion_ids == tokenizer.eos_token_id) & completion_pad_mask  # [B*num_gen, R]
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1) - 1, dtype=torch.long, device=args.device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        completion_mask = ((torch.arange(is_eos.size(1), device=args.device).expand(is_eos.size(0), -1) <= eos_idx.unsqueeze(1)) & completion_pad_mask).int()  # [B*num_gen, R]

        # Reuse the fixed rollout and old log-probabilities for several policy
        # updates.  With one update the first ratio is ~1 for the torch engine,
        # which makes clipping ablations largely uninformative.
        policy_output = None
        aux_loss = torch.tensor(0.0, device=args.device)
        per_token_logps = None
        rollout_logprob_mae = None
        rollout_ratio_mean = None
        for policy_epoch in range(args.policy_update_epochs):
            with autocast_ctx:
                res = model(outputs, attention_mask=full_mask)
                aux_loss = res.aux_loss if lm_config.use_moe else torch.tensor(0.0, device=args.device)
                per_token_logps = F.log_softmax(res.logits[:, :-1, :].float(), dim=-1).gather(
                    2, outputs[:, 1:].unsqueeze(-1)
                ).squeeze(-1).gather(1, logp_pos)
                if policy_epoch == 0:
                    valid_tokens = completion_mask.sum().clamp(min=1)
                    initial_log_ratio = per_token_logps.float() - old_per_token_logps
                    rollout_logprob_mae = (
                        initial_log_ratio.abs() * completion_mask
                    ).sum() / valid_tokens
                    rollout_ratio_mean = (
                        initial_log_ratio.exp() * completion_mask
                    ).sum() / valid_tokens
                policy_output = compute_policy_loss(
                    loss_type=args.loss_type,
                    current_logps=per_token_logps,
                    old_logps=old_per_token_logps,
                    reference_logps=ref_per_token_logps,
                    advantages=advantages,
                    completion_mask=completion_mask,
                    beta=args.beta,
                    grpo_epsilon=args.epsilon,
                    cispo_epsilon_high=args.epsilon_high,
                    dapo_epsilon_low=args.dapo_epsilon_low,
                    dapo_epsilon_high=args.dapo_epsilon_high,
                    gspo_epsilon_low=args.gspo_epsilon_low,
                    gspo_epsilon_high=args.gspo_epsilon_high,
                )
                token_mean_scale = (
                    distributed_token_mean_scale(completion_mask)
                    if args.loss_type in {"cispo", "dapo"} else 1.0
                )
                loss = (policy_output.loss * token_mean_scale + aux_loss) / args.accumulation_steps
            loss.backward()

            last_micro_step = ((epoch * iters + step - 1) * args.policy_update_epochs
                               + policy_epoch + 1)
            if last_micro_step % args.accumulation_steps == 0:
                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

        if step % args.log_interval == 0 or step == iters:
            policy_loss_val = policy_output.policy_loss.item()
            current_aux_loss = aux_loss.item()
            avg_reward_val = rewards.mean().item()
            avg_len_val = completion_mask.sum(dim=1).float().mean().item()
            kl_ref_val = policy_output.approx_kl.item()
            advantages_mean_val = advantages.mean().item()
            advantages_std_val = advantages.std().item()
            current_lr = optimizer.param_groups[0]['lr']
            clip_fraction = policy_output.clip_fraction.item()
            ratio_mean = policy_output.ratio_mean.item()
            ratio_std = policy_output.ratio_std.item()

            Logger(f'Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}), '
                   f'Algorithm:{args.loss_type.upper()}, Reward:{avg_reward_val:.4f}, KL_k3:{kl_ref_val:.6f}, '
                   f'Adv Std: {advantages_std_val:.4f}, Adv Mean: {advantages_mean_val:.4f}, '
                   f'Actor Loss:{policy_loss_val:.4f}, ClipFrac:{clip_fraction:.4f}, '
                   f'Ratio:{ratio_mean:.4f}±{ratio_std:.4f}, Avg Response Len:{avg_len_val:.2f}, LR:{current_lr:.8f}')

            metrics = {
                "reward": avg_reward_val,
                "kl_k3": kl_ref_val,
                "group_reward_std": grouped_rewards.std(dim=1, unbiased=False).mean().item(),
                "advantages_std": advantages_std_val,
                "advantages_mean": advantages_mean_val,
                "policy_loss": policy_loss_val,
                "aux_loss": current_aux_loss,
                "clip_fraction": clip_fraction,
                "ratio_mean": ratio_mean,
                "ratio_std": ratio_std,
                "rollout_logprob_mae": rollout_logprob_mae.item(),
                "rollout_ratio_mean": rollout_ratio_mean.item(),
                "avg_response_len": avg_len_val,
                "learning_rate": current_lr,
            }
            if is_main_process():
                metric_logger.log(metrics, step=epoch * iters + step)

            if wandb and is_main_process():
                wandb.log(metrics)

        if (step % args.save_interval == 0 or step == iters) and is_main_process():
            model.eval()
            moe_suffix = '_moe' if lm_config.use_moe else ''
            ckp = f'{args.save_dir}/{args.save_weight}_{lm_config.hidden_size}{moe_suffix}.pth'
            raw_model = model.module if isinstance(model, DistributedDataParallel) else model
            raw_model = getattr(raw_model, '_orig_mod', raw_model)
            state_dict = raw_model.state_dict()
            torch.save({k: v.half().cpu() for k, v in state_dict.items()}, ckp)
            lm_checkpoint(lm_config, weight=args.save_weight, model=model, optimizer=optimizer, 
                         epoch=epoch, step=step, wandb=wandb, save_dir='../checkpoints', scheduler=scheduler)
            model.train()
            del state_dict

        # A stale remote rollout policy invalidates controlled on-policy
        # comparisons.  The torch engine only updates a pointer and is cheap.
        if step % args.rollout_sync_interval == 0 or step == iters:
            rollout_engine.update_policy(model)

        del prompt_inputs, outputs, completion_ids, per_token_logps, ref_per_token_logps
        del completions, rewards, grouped_rewards, advantages, completion_mask, completion_pad_mask, prompt_lens, logp_pos

    if last_micro_step and last_micro_step % args.accumulation_steps != 0:
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniMind GRPO (Group Relative Policy Optimization)")
    parser.add_argument("--save_dir", type=str, default="../out", help="模型保存目录")
    parser.add_argument('--save_weight', default='grpo', type=str, help="保存权重的前缀名")
    parser.add_argument("--epochs", type=int, default=1, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=2, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=3e-7, help="初始学习率")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="RL AdamW权重衰减；默认0避免零advantage时策略漂移")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="训练设备")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16", "float16"], help="混合精度类型")
    parser.add_argument("--num_workers", type=int, default=8, help="数据加载线程数")
    parser.add_argument("--accumulation_steps", type=int, default=1, help="梯度累积步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--log_interval", type=int, default=1, help="日志打印间隔")
    parser.add_argument("--save_interval", type=int, default=10, help="模型保存间隔")
    parser.add_argument('--hidden_size', default=768, type=int, help="隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="隐藏层数量")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE架构（0=否，1=是）")
    parser.add_argument('--max_seq_len', default=768, type=int, help="Prompt最大长度")
    parser.add_argument("--max_gen_len", type=int, default=1024, help="生成的最大长度")
    parser.add_argument("--data_path", type=str, default="../dataset/rlaif.jsonl", help="RLAIF数据路径")
    parser.add_argument("--num_generations", type=int, default=6, help="每个prompt生成的样本数")
    parser.add_argument("--beta", type=float, default=0.1, help="KL惩罚系数")
    parser.add_argument("--loss_type", type=str, default="cispo", choices=["grpo", "cispo", "dapo", "gspo"], help="策略优化目标")
    parser.add_argument("--epsilon", type=float, default=0.2, help="GRPO的PPO clip epsilon")
    parser.add_argument("--epsilon_high", type=float, default=5.0, help="CISPO epsilon_high_IS；实际ratio上界为1+该值")
    parser.add_argument("--dapo_epsilon_low", type=float, default=0.2, help="DAPO非对称下裁剪")
    parser.add_argument("--dapo_epsilon_high", type=float, default=0.28, help="DAPO Clip-Higher上裁剪")
    parser.add_argument("--gspo_epsilon_low", type=float, default=3e-4, help="GSPO序列比率下裁剪")
    parser.add_argument("--gspo_epsilon_high", type=float, default=4e-4, help="GSPO序列比率上裁剪")
    parser.add_argument("--policy_update_epochs", type=int, default=1, help="每批rollout复用的策略更新轮数；算法对比建议2~4")
    parser.add_argument("--rollout_sync_interval", type=int, default=1, help="多少个训练batch同步一次rollout策略")
    parser.add_argument("--metrics_path", type=str, default="../out/metrics/grpo_family.jsonl", help="本地可审计JSONL指标")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument('--from_weight', default='full_sft', type=str, help="基于哪个权重训练")
    parser.add_argument("--reward_model_path", type=str, default="../../internlm2-1_8b-reward", help="Reward模型路径")
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1], help="是否自动检测&续训（0=否，1=是）")
    parser.add_argument("--use_wandb", action="store_true", help="是否使用wandb")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-GRPO", help="wandb项目名")
    parser.add_argument("--use_compile", default=0, type=int, choices=[0, 1], help="是否使用torch.compile加速（0=否，1=是）")
    parser.add_argument("--debug_mode", action="store_true", help="是否打印训练调试采样")
    parser.add_argument("--debug_interval", type=int, default=20, help="debug模式下每隔多少step打印一次采样")
    parser.add_argument("--thinking_ratio", type=float, default=0.9, help="按概率开启thinking（0.0~1.0）")
    parser.add_argument("--rollout_engine", type=str, default="torch", choices=["torch", "sglang"], help="rollout引擎类型")
    parser.add_argument("--sglang_base_url", type=str, default="http://localhost:8998", help="SGLang服务器URL")
    parser.add_argument("--sglang_model_path", type=str, default="../model", help="SGLang tokenizer路径")
    parser.add_argument("--sglang_shared_path", type=str, default="./sglang_ckpt_grpo", help="SGLang共享存储路径")
    args = parser.parse_args()
    if args.policy_update_epochs < 1 or args.rollout_sync_interval < 1:
        parser.error("policy_update_epochs and rollout_sync_interval must be >= 1")
    if args.num_generations < 2 or args.batch_size < 1:
        parser.error("num_generations must be >= 2 and batch_size must be >= 1")
    if args.epsilon_high < 0:
        parser.error("epsilon_high must be >= 0")
    if args.weight_decay < 0:
        parser.error("weight_decay must be >= 0")

    # ========== 1. 初始化环境和随机种子 ==========
    local_rank = init_distributed_mode()
    if dist.is_initialized(): args.device = f"cuda:{local_rank}"
    process_rank = dist.get_rank() if dist.is_initialized() else 0
    setup_seed(args.seed + process_rank)
    
    # ========== 2. 配置目录、模型参数、检查ckp ==========
    os.makedirs(args.save_dir, exist_ok=True)
    metric_logger = JsonlMetricLogger(
        args.metrics_path if is_main_process() else None,
        run_config={
            "algorithm": args.loss_type,
            "data_path": args.data_path,
            "seed": args.seed,
            "num_generations": args.num_generations,
            "policy_update_epochs": args.policy_update_epochs,
            "beta": args.beta,
            "weight_decay": args.weight_decay,
            "hidden_size": args.hidden_size,
            "num_hidden_layers": args.num_hidden_layers,
            "use_moe": bool(args.use_moe),
        },
    )
    lm_config = MiniMindConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers,
                               max_seq_len=args.max_seq_len + args.max_gen_len, use_moe=bool(args.use_moe))
    ckp_data = lm_checkpoint(lm_config, weight=args.save_weight, save_dir='../checkpoints') if args.from_resume==1 else None
    
    # ========== 3. 设置混合精度 ==========
    device_type = "cuda" if "cuda" in args.device else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    autocast_ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast(dtype=dtype)
    
    # ========== 4. 配wandb ==========
    wandb = None
    if args.use_wandb and is_main_process():
        import swanlab as wandb
        wandb_id = ckp_data.get('wandb_id') if ckp_data else None
        resume = 'must' if wandb_id else None
        wandb_run_name = f"MiniMind-GRPO-Epoch-{args.epochs}-BS-{args.batch_size}-LR-{args.learning_rate}"
        wandb.init(project=args.wandb_project, name=wandb_run_name, id=wandb_id, resume=resume)
    
    # ========== 5. 初始化模型和数据 ==========
    base_weight = args.from_weight
    # Policy模型
    model, tokenizer = init_model(lm_config, base_weight, device=args.device)
    # Reference模型
    ref_model, _ = init_model(lm_config, base_weight, device=args.device)
    ref_model = ref_model.eval().requires_grad_(False)
    # Reward模型
    reward_model = LMForRewardModel(args.reward_model_path, device=args.device, dtype=torch.float16)
    # Rollout引擎（可插拔替换，只负责 policy 推理）
    rollout_engine = create_rollout_engine(
        engine_type=args.rollout_engine,
        policy_model=model,
        tokenizer=tokenizer,
        device=args.device,
        autocast_ctx=autocast_ctx,
        sglang_base_url=args.sglang_base_url,
        sglang_model_path=args.sglang_model_path,
        sglang_shared_path=args.sglang_shared_path,
    )
    # 数据和优化器
    train_ds = RLAIFDataset(args.data_path, tokenizer, max_length=lm_config.max_seq_len, thinking_ratio=args.thinking_ratio)
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    loader_for_count = DataLoader(train_ds, batch_size=args.batch_size, sampler=train_sampler)
    iters = len(loader_for_count)
    total_micro_steps = iters * args.policy_update_epochs * args.epochs
    total_optimizer_steps = math.ceil(total_micro_steps / args.accumulation_steps)
    scheduler = CosineAnnealingLR(optimizer, T_max=total_optimizer_steps, eta_min=args.learning_rate / 10)
    
    # ========== 6. 从ckp恢复状态 ==========
    start_epoch, start_step = 0, 0
    if ckp_data:
        model.load_state_dict(ckp_data['model'])
        optimizer.load_state_dict(ckp_data['optimizer'])
        scheduler.load_state_dict(ckp_data['scheduler'])
        start_epoch = ckp_data['epoch']
        start_step = ckp_data.get('step', 0)
    
    # ========== 7. 编译和分布式包装 ==========
    if args.use_compile == 1:
        model = torch.compile(model)
        Logger('torch.compile enabled')
        rollout_engine.update_policy(model)
    if dist.is_initialized():
        model = DistributedDataParallel(model, device_ids=[local_rank])
    rollout_engine.update_policy(model)
    
    # ========== 8. 开始训练 ==========
    for epoch in range(start_epoch, args.epochs):
        train_sampler and train_sampler.set_epoch(epoch)
        setup_seed(args.seed + epoch * max(dist.get_world_size() if dist.is_initialized() else 1, 1) + process_rank)
        indices = torch.randperm(len(train_ds)).tolist()
        skip = start_step if (epoch == start_epoch and start_step > 0) else 0
        batch_sampler = SkipBatchSampler(train_sampler or indices, args.batch_size, skip)
        loader = DataLoader(train_ds, batch_sampler=batch_sampler, num_workers=args.num_workers, pin_memory=True)
        if skip > 0: 
            Logger(f'Epoch [{epoch + 1}/{args.epochs}]: 跳过前{start_step}个step，从step {start_step + 1}开始')
            grpo_train_epoch(epoch, loader, len(loader) + skip, rollout_engine, ref_model, reward_model, start_step, wandb, use_sglang = (args.rollout_engine == "sglang"))
        else:
            grpo_train_epoch(epoch, loader, len(loader), rollout_engine, ref_model, reward_model, 0, wandb, use_sglang = (args.rollout_engine == "sglang"))
    
    # ========== 9. 清理分布进程 ==========
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()
