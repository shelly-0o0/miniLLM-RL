from torch.utils.data import Dataset
import torch
import json
import os
import random
from datasets import load_dataset, Features, Sequence, Value
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def pre_processing_chat(conversations, add_system_ratio=0.2):
    """对对话数据做轻量预处理。

    - 如果样本里包含工具调用相关字段，则直接返回，避免破坏 tool-use 结构。
    - 普通对话按概率补一个 system 角色，增强模型对系统提示词的鲁棒性。
    """
    # tool use 数据完整保留不做处理
    if any(conv.get('tools') for conv in conversations): return conversations

    SYSTEM_PROMPTS = [
        "你是一个知识丰富的AI，尽力为用户提供准确的信息。",
        "你是minimind，一个小巧但有用的语言模型。",
        "你是一个专业的AI助手，请提供有价值的回答。",
        "你是minimind，请尽力帮助用户解决问题。",
        "你是一个可靠的AI，请给出准确的回答。",
        "You are a helpful AI assistant.",
        "You are minimind, a lightweight intelligent assistant.",
        "You are a friendly chatbot. Please answer the user's questions carefully.",
        "You are a knowledgeable AI. Try your best to provide accurate information.",
        "You are minimind, a small but useful language model."
    ]
    # 概率性添加system
    if conversations[0].get('role') != 'system':
        if random.random() < add_system_ratio:
            return [{'role': 'system', 'content': random.choice(SYSTEM_PROMPTS)}] + conversations
    return conversations

def post_processing_chat(prompt_content, empty_think_ratio=0.2):
    """对拼好的 prompt 做后处理。"""
    # 以 80% 的概率移除空思考标签，减少训练时出现无意义的 <think></think>
    if '<think>\n\n</think>\n\n' in prompt_content and random.random() > empty_think_ratio:
        prompt_content = prompt_content.replace('<think>\n\n</think>\n\n', '')
    return prompt_content

class PretrainDataset(Dataset):
    """预训练数据集。

    输入是纯文本 `text`，输出是自回归语言模型训练所需的
    `input_ids` 和 `labels`。
    """
    def __init__(self, data_path, tokenizer, max_length=512):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = load_dataset('json', data_files=data_path, split='train')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        # 先截断文本，再补上 BOS/EOS，保证总长度稳定在 max_length 内。
        tokens = self.tokenizer(str(sample['text']), add_special_tokens=False, max_length=self.max_length - 2, truncation=True).input_ids
        tokens = [self.tokenizer.bos_token_id] + tokens + [self.tokenizer.eos_token_id]
        # padding 部分填充到固定长度，便于 batch 训练。
        input_ids = tokens + [self.tokenizer.pad_token_id] * (self.max_length - len(tokens))
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        # 预训练任务里，pad token 不参与 loss，因此其 label 置为 -100。
        labels = input_ids.clone()
        labels[input_ids == self.tokenizer.pad_token_id] = -100
        return input_ids, labels


class SFTDataset(Dataset):
    """监督微调数据集（SFT）。

    读取多轮对话样本，将对话模板拼成训练文本，并且只对
    assistant 的回复部分计算 loss。
    """
    def __init__(self, jsonl_path, tokenizer, max_length=1024):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        # 明确每条消息允许出现的字段，避免 JSON 结构不稳定时解析出错。
        features = Features({'conversations': [{'role': Value('string'), 'content': Value('string'), 'reasoning_content': Value('string'), 'tools': Value('string'), 'tool_calls': Value('string')}]})
        self.samples = load_dataset('json', data_files=jsonl_path, split='train', features=features)
        # 这里的 BOS/EOS 用于在 token 序列中定位 assistant 回复区间。
        self.bos_id = tokenizer(f'{tokenizer.bos_token}assistant\n', add_special_tokens=False).input_ids
        self.eos_id = tokenizer(f'{tokenizer.eos_token}\n', add_special_tokens=False).input_ids

    def __len__(self):
        return len(self.samples)

    def create_chat_prompt(self, conversations):
        # 复制一份消息，避免在原始样本上直接修改字段。
        messages = []
        tools = None
        for message in conversations:
            message = dict(message)
            # system 消息里若携带 tools，则提取出来交给 chat template。
            if message.get("role") == "system" and message.get("tools"):
                tools = json.loads(message["tools"]) if isinstance(message["tools"], str) else message["tools"]
            # tool_calls 可能以字符串形式存储，这里统一转成对象。
            if message.get("tool_calls") and isinstance(message["tool_calls"], str):
                message["tool_calls"] = json.loads(message["tool_calls"])
            messages.append(message)
        # 交给 tokenizer 的 chat template 统一拼接成最终 prompt。
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            tools=tools
        )

    def generate_labels(self, input_ids):
        # 默认全部忽略，只把 assistant 回复对应的 token 作为监督目标。
        labels = [-100] * len(input_ids)
        i = 0
        while i < len(input_ids):
            # 找到一个 assistant 回复段的起始位置。
            if input_ids[i:i + len(self.bos_id)] == self.bos_id:
                start = i + len(self.bos_id)
                end = start
                # 往后找到该回复段对应的 EOS。
                while end < len(input_ids):
                    if input_ids[end:end + len(self.eos_id)] == self.eos_id:
                        break
                    end += 1
                # assistant 回复段内的 token 都参与 loss。
                for j in range(start, min(end + len(self.eos_id), self.max_length)):
                    labels[j] = input_ids[j]
                i = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)
            else:
                i += 1
        return labels

    def __getitem__(self, index):
        sample = self.samples[index]
        # 预处理 -> 拼 prompt -> 可选清理空 think 标签 -> tokenize。
        conversations = pre_processing_chat(sample['conversations'])
        prompt = self.create_chat_prompt(conversations)
        prompt = post_processing_chat(prompt)
        input_ids = self.tokenizer(prompt).input_ids[:self.max_length]
        # 不足 max_length 的部分补 pad。
        input_ids += [self.tokenizer.pad_token_id] * (self.max_length - len(input_ids))
        labels = self.generate_labels(input_ids)
        # # === 调试打印 ===
        # print(f"\n--- Sample {index} ---")
        # for i, (x, y) in enumerate(zip(input_ids[:-1], labels[1:])):
        #     print(f"{i:3d}: X={self.tokenizer.decode([x])!r:16s} ---> Y={self.tokenizer.decode([input_ids[i+1]])!r:16s} label={y}")
        # # ================
        return torch.tensor(input_ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long)


class DPODataset(Dataset):
    """DPO 数据集。

    每条样本包含 `chosen` 和 `rejected` 两条对话，分别构造 token 序列
    和 loss mask，用于偏好优化训练。
    """
    def __init__(self, file_path, tokenizer, max_length=4096):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.padding = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        # 用于识别 assistant 回复区间，与 SFT 的标签生成逻辑保持一致。
        self.bos_id = tokenizer(f'{tokenizer.bos_token}assistant\n', add_special_tokens=False).input_ids
        self.eos_id = tokenizer(f'{tokenizer.eos_token}\n', add_special_tokens=False).input_ids
        self.samples = load_dataset('json', data_files=file_path, split='train')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        # chosen / rejected 都是完整对话列表，分别代表优选答案和劣选答案。
        chosen = sample['chosen']
        rejected = sample['rejected']
        chosen_prompt = self.tokenizer.apply_chat_template(
            chosen, tokenize=False, add_generation_prompt=False
        )
        chosen_prompt = post_processing_chat(chosen_prompt)

        rejected_prompt = self.tokenizer.apply_chat_template(
            rejected, tokenize=False, add_generation_prompt=False
        )
        rejected_prompt = post_processing_chat(rejected_prompt)
        chosen_encoding = self.tokenizer(
            chosen_prompt, truncation=True, max_length=self.max_length, padding='max_length'
        )
        rejected_encoding = self.tokenizer(
            rejected_prompt, truncation=True, max_length=self.max_length, padding='max_length'
        )

        chosen_input_ids = chosen_encoding['input_ids']
        # 只让 assistant 回复部分参与 loss 统计。
        chosen_loss_mask = self.generate_loss_mask(chosen_input_ids)

        rejected_input_ids = rejected_encoding['input_ids']
        rejected_loss_mask = self.generate_loss_mask(rejected_input_ids)
        # DPO / 偏好类训练通常使用 next-token 形式，所以这里做 shift。
        x_chosen = torch.tensor(chosen_input_ids[:-1], dtype=torch.long)
        y_chosen = torch.tensor(chosen_input_ids[1:], dtype=torch.long)
        mask_chosen = torch.tensor(chosen_loss_mask[1:], dtype=torch.long)
        x_rejected = torch.tensor(rejected_input_ids[:-1], dtype=torch.long)
        y_rejected = torch.tensor(rejected_input_ids[1:], dtype=torch.long)
        mask_rejected = torch.tensor(rejected_loss_mask[1:], dtype=torch.long)

        return {
            'x_chosen': x_chosen,
            'y_chosen': y_chosen,
            'mask_chosen': mask_chosen,
            'x_rejected': x_rejected,
            'y_rejected': y_rejected,
            'mask_rejected': mask_rejected
        }

    def generate_loss_mask(self, input_ids):
        # 默认不计算 loss，只标记 assistant 回复段对应的位置为 1。
        loss_mask = [0] * len(input_ids)
        i = 0
        while i < len(input_ids):
            if input_ids[i:i + len(self.bos_id)] == self.bos_id:
                start = i + len(self.bos_id)
                end = start
                while end < len(input_ids):
                    if input_ids[end:end + len(self.eos_id)] == self.eos_id:
                        break
                    end += 1
                # EOS 也一并纳入监督范围，避免回复边界被截断。
                for j in range(start, min(end + len(self.eos_id), self.max_length)):
                    loss_mask[j] = 1
                i = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)
            else:
                i += 1
        return loss_mask


class RLAIFDataset(Dataset):
    """RLAIF / PPO / GRPO 使用的提示词数据集。

    这个数据集不直接返回监督答案，只返回 prompt，让 rollout 阶段
    把模型自己的生成结果拿去算 reward。
    """
    def __init__(self, jsonl_path, tokenizer, max_length=1024, thinking_ratio=0.5):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        # 按概率开启 thinking，帮助模型在训练时覆盖不同的输出格式。
        self.thinking_ratio = thinking_ratio
        self.samples = load_dataset('json', data_files=jsonl_path, split='train')
        # 这两个字段在当前实现里未直接使用，保留是为了与其他数据集接口风格一致。
        self.bos_id = tokenizer(f'{tokenizer.bos_token}assistant', add_special_tokens=False).input_ids
        self.eos_id = tokenizer(f'{tokenizer.eos_token}', add_special_tokens=False).input_ids

    def __len__(self):
        return len(self.samples)

    def create_chat_prompt(self, conversations):
        # 先做基础清洗，再决定是否打开 thinking 模式。
        conversations = pre_processing_chat(conversations)
        use_thinking = random.random() < self.thinking_ratio
        # 只保留上下文部分，最后一轮留给模型自己生成。
        return self.tokenizer.apply_chat_template(
            conversations[:-1],
            tokenize=False,
            open_thinking=use_thinking,
            add_generation_prompt=True
        )

    def __getitem__(self, index):
        sample = self.samples[index]
        prompt = self.create_chat_prompt(sample['conversations'])

        # rollout 阶段只需要 prompt，answer 留空位以适配统一的数据接口。
        return {
            'prompt': prompt,
            'answer': ""
        }


class AgentRLDataset(Dataset):
    """Agent 强化学习数据集。

    每条样本包含对话 messages、工具定义 tools 和 ground-truth `gt`，
    供工具调用场景下的 rollout 与 reward 计算使用。
    """
    def __init__(self, jsonl_path, tokenizer, max_length=1024):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []
        # JSONL 一行一个样本，适合大规模数据逐行读取。
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                self.samples.append(json.loads(line.strip()))

    def __len__(self):
        return len(self.samples)

    def parse_conversations(self, conversations):
        # 把 tools 从 system 消息里提取出来，方便后续单独传给 chat template。
        messages = []
        tools = None
        for message in conversations:
            message = dict(message)
            if message.get("role") == "system" and message.get("tools"):
                tools = json.loads(message["tools"]) if isinstance(message["tools"], str) else message["tools"]
            messages.append(message)
        # 最后一条通常是需要单独监督/评估的答案，因此从 prompt 中去掉。
        return messages[:-1], tools

    def __getitem__(self, index):
        sample = self.samples[index]
        messages, tools = self.parse_conversations(sample['conversations'])
        required_tools = list(sample.get('required_tools') or [])
        # The upstream verifiable Agent subset is calculator-only but predates
        # an explicit required-tools field.  Preserve its semantics instead of
        # allowing a valid distractor call plus a guessed answer to pass.
        available_names = {
            tool.get('function', {}).get('name') for tool in (tools or [])
        }
        if not required_tools and sample.get('gt') and 'calculate_math' in available_names:
            required_tools = ['calculate_math']
        # gt 是 reward 计算时使用的标准答案或关键事实。
        return {
            'messages': messages,
            'tools': tools,
            'required_tools': required_tools,
            'gt': sample['gt'],
        }


if __name__ == "__main__":
    pass
