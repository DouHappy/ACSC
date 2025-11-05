from functools import partial
import os
import json
from tqdm import tqdm
from typing import Any, Optional, Callable, Sequence
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from pathlib import Path

from exp.analyzer import get_normal_token_coresponding, find_first_diff, first_diff_p2minus_k, normal_token_p2minus_k

tokenizer:Optional[AutoTokenizer] = None
model:Optional[AutoModel] = None

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def load_model(model_path, device_ids = '0') -> None:
    os.environ['CUDA_VISIBLE_DEVICES'] = device_ids
    global tokenizer, model
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModel.from_pretrained(
        model_path, 
        output_attentions=True,
        output_hidden_states=True,
        device_map='auto', 
        torch_dtype=torch.bfloat16,
        ).to('cuda')

def load_input(message_path, start_id=None, max_len=None):
    with open(message_path, "r") as f:
        datas = json.load(f)
    
    if start_id is not None:
        datas['metadata']['result'] = datas['metadata']['result'][start_id:]
    if max_len is not None:
        datas['metadata']['result'] = datas['metadata']['result'][:max_len]
    return datas

def get_message(datas):
    messages: list[Any] = []
    for d in datas['metadata']['result']:
        message = d['prompt']
        message.append(
            {
                "role": "assistant",
                "content": d['target'],
            }
        )
        messages.append(message)
    
    return messages

def get_edit_operations(src_tokens, tgt_tokens):
    m, n = len(src_tokens), len(tgt_tokens)
    
    # dp[i][j] 表示 src[:i] 到 tgt[:j] 的最小编辑距离
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    # 初始化边界
    for i in range(m + 1):
        dp[i][0] = i  # 删除所有 src 字符
    for j in range(n + 1):
        dp[0][j] = j  # 插入所有 tgt 字符
    
    # 填充 dp 表
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if src_tokens[i - 1] == tgt_tokens[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = min(
                    dp[i - 1][j] + 1,    # 删除
                    dp[i][j - 1] + 1,    # 插入
                    dp[i - 1][j - 1] + 1 # 替换
                )
    
    # 回溯获取操作序列
    operations: list[Any] = []
    i, j = m, n
    while i > 0 or j > 0:
        if i > 0 and j > 0 and src_tokens[i - 1] == tgt_tokens[j - 1]:
            # 相等，无操作，直接回退
            i -= 1
            j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
            # 替换
            operations.append((i - 1, j - 1, "mod"))
            i -= 1
            j -= 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            # 删除 src[i-1]
            operations.append((i - 1, j, "del"))  # 注意：j 保持不变，因为目标序列该位置未被消费
            i -= 1
        elif j > 0 and dp[i][j] == dp[i][j - 1] + 1:
            # 插入 tgt[j-1]
            operations.append((i, j - 1, "add"))  # i 不变，因为源序列未消费新 token
            j -= 1
        else:
            break  # 安全退出
    
    # 由于回溯是从后往前，所以反转操作序列
    operations.reverse()
    return operations


def _str2token_list(text: str) -> list[str]:
    if tokenizer is None:
        raise RuntimeError("Tokenizer is not loaded. Call load_model first.")
    token_ids = tokenizer.encode(text)
    return [tokenizer.decode([token_id]) for token_id in token_ids]

def statistic_attn(
        datas, 
        save_path, 
        batchsize=16, 
        process_start_id=0, 
        input_key="source", 
        output_key="target",
    ):
    # 获取message
    global tokenizer, model
    messages: list[Any] = get_message(datas)
    # 获取基本的输入
    sources: list[str] = [d[input_key] for d in datas['metadata']['result']]
    predictions: list[str] = [d[output_key] for d in datas['metadata']['result']]
    
    # 计算instruct的长度，只存剩下的atten分布
    input_str0: str = tokenizer.apply_chat_template(messages[0], tokenize=False)
    instruct: str = input_str0[:input_str0.find(sources[0])]
    instruct_token_len: int = len(tokenizer.encode(instruct))
    print(f"instruct_token_len = {instruct_token_len}")
    
    # 创建保存目录
    os.makedirs(save_path, exist_ok=True)
    
    # 存储所有attention数据的列表
    all_metadata = []
    
    for start_id in tqdm(range(process_start_id, len(messages), batchsize), desc="计算Atten"):
        end_id: int = min(start_id + batchsize, len(messages))
        
        input_ids = tokenizer.apply_chat_template(messages[start_id:end_id], 
                                                  padding=True, 
                                                  return_tensors="pt").to('cuda')
        
        # 获取模型输出（包含attention）
        with torch.no_grad():
            outputs = model(input_ids=input_ids, output_attentions=True)
        
        # 处理attention数据
        attentions = outputs.attentions  # tuple of (batch_size, num_heads, seq_len, seq_len)
        
        if start_id == 0:
            print(f"attention layers: {len(attentions)}")
            print(f"attention shape per layer: {attentions[0].shape}")
        
        # 提取instruction后面的部分 - 修正索引方式
        batch_attentions = []
        for layer_idx, layer_attn in enumerate(attentions):
            # layer_attn shape: (batch_size, num_heads, seq_len, seq_len)
            # 只保留instruction之后的token对instruction之后token的attention
            cropped_attn = layer_attn[:, :, instruct_token_len:, instruct_token_len:]
            batch_attentions.append(cropped_attn.to(torch.float32).cpu())
        
        # 转换为numpy数组: (num_layers, batch_size, num_heads, cropped_seq_len, cropped_seq_len)
        batch_attentions = np.stack(batch_attentions, axis=0)
        
        # 保存对应的元数据
        batch_metadata = {
            'sources': sources[start_id:end_id],
            'predictions': predictions[start_id:end_id],
            'start_id': start_id,
            'end_id': end_id
        }
        all_metadata.append(batch_metadata)
        
        # 可选：每个batch单独保存（适合大数据集）
        batch_save_path = os.path.join(save_path, f'attention_batch_{start_id}_{end_id}.npz')
        np.savez(
            batch_save_path,
            attentions=batch_attentions,
            sources=np.array(sources[start_id:end_id]),
            predictions=np.array(predictions[start_id:end_id]),
            instruct_token_len=instruct_token_len,
            start_id=start_id,
            end_id=end_id
        )
        
        print(f"Saved batch {start_id}-{end_id} to {batch_save_path}")
    
    # 保存配置信息
    config_path = os.path.join(save_path, 'config.json')
    config = {
        'instruct_token_len': instruct_token_len,
        'total_samples': len(messages),
        'num_layers': len(attentions),
        'num_heads': attentions[0].shape[1],
        'batchsize': batchsize,
        'save_path': save_path
    }
    
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    print(f"✅ 配置信息保存到: {config_path}")
    print(f"✅ 总共处理了 {len(messages)} 个样本")
    
    return config


def realtime_pre_extract(
    datas: dict[str, Any],
    exp_position_functions: dict[str, Any],
    save_path: str,
    batchsize: int = 16,
    process_start_id: int = 0,
    add_instruction: bool = False,
    crop_shift: int = 0,
    input_key: str = "source",
    output_key: str = "target",
) -> dict[str, Any]:
    """
    实时推理attention并提取特征，避免中间磁盘I/O。
    """
    global tokenizer, model
    if tokenizer is None or model is None:
        raise RuntimeError("Tokenizer or model is not loaded. Call load_model first.")
    model.eval()

    messages: list[Any] = get_message(datas)
    sources: list[str] = [d[input_key] for d in datas['metadata']['result']]
    targets: list[str] = [d[output_key] for d in datas['metadata']['result']]

    input_str0: str = tokenizer.apply_chat_template(messages[0], tokenize=False)
    instruct: str = input_str0[:input_str0.find(sources[0])]
    instruct_token_len: int = len(tokenizer.encode(instruct))
    print(f"instruct_token_len = {instruct_token_len}")

    src_tgt_tokens = [
        (_str2token_list(src), _str2token_list(tgt))
        for src, tgt in zip(sources, targets)
    ]
    exp_positions = [
        [
            exp_position_function(src_tokens, tgt_tokens)
            for src_tokens, tgt_tokens in src_tgt_tokens
        ]
        for exp_position_function in exp_position_functions.values()
    ]
    exp_names = list(exp_position_functions.keys())
    exp_attn_matrix: dict[str, list[tuple[int, int, np.ndarray]]] = {
        exp_name: [] for exp_name in exp_names
    }
    # 创建instruct的attention分布
    for exp_name in exp_names:
        exp_attn_matrix[f"{exp_name}_ins"] = []

    total_samples = len(messages)
    num_layers = 0
    num_heads = 0
    for start_id in tqdm(range(process_start_id, total_samples, batchsize), desc="实时提取特征"):
        end_id: int = min(start_id + batchsize, total_samples)
        batch_messages = messages[start_id:end_id]
        input_ids = tokenizer.apply_chat_template(
            batch_messages,
            padding=True,
            return_tensors="pt"
        ).to('cuda')

        with torch.no_grad():
            outputs = model(input_ids=input_ids, output_attentions=True)

        attentions = outputs.attentions
        if not num_layers:
            num_layers = len(attentions)
            num_heads = attentions[0].shape[1] if attentions else 0
        cropped_layers = []
        instruct_attn = []
        for layer_attn in attentions:
            # 为了计算更快 临时裁剪一下
            cropped = layer_attn[:, :, instruct_token_len - crop_shift:, instruct_token_len - crop_shift:]
            cropped_layers.append(cropped.to(torch.float32).cpu())
            if add_instruction:
                instruct_attn.append(
                    layer_attn[:, :, instruct_token_len - crop_shift:, :instruct_token_len].sum(dim=-1, keepdims=True).to(torch.float32).cpu()
                )
        batch_attentions = torch.stack(cropped_layers, dim=0).numpy()
        if add_instruction:
            instruct_attn = torch.stack(instruct_attn, dim=0).numpy()
        # batch_attentions = torch.stack([a.to(torch.float32).cpu() for a in attentions]).numpy()
        for batch_offset, sample_idx in enumerate(range(start_id, end_id)):
            # print(f"processing sample {sample_idx}")
            # print(f"message: {messages[sample_idx]}")
            attn_matrix = batch_attentions[:, batch_offset]
            for eid, e_pos_list in enumerate(exp_positions):
                for pos_idx, (qid, kids) in enumerate(e_pos_list[sample_idx]):
                    if qid is None:
                        continue
                    # print(f"qid and kids: {qid}, {kids}")
                    kid_plus_instruct = [k+crop_shift for k in kids]
                    qid_attn_matrix = attn_matrix[:, :, qid+crop_shift, kid_plus_instruct]
                    if add_instruction == True:
                        qid_attn_matrix = np.concatenate(
                            [instruct_attn[:, batch_offset, :, qid+crop_shift, :], qid_attn_matrix],
                            axis=-1
                        )

                    exp_attn_matrix[exp_names[eid]].append(
                        (sample_idx, pos_idx, qid_attn_matrix)
                    )
                    

        del batch_attentions
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    save_dir = Path(save_path)
    save_dir.mkdir(parents=True, exist_ok=True)
    meta_info = {
        "instruct_token_len": instruct_token_len,
        "total_samples": total_samples,
        "num_layers": num_layers,
        "num_heads": num_heads,
        "batchsize": batchsize,
        "save_path": str(save_dir)
    }

    for exp_name in exp_names:
        ordered = sorted(exp_attn_matrix[exp_name], key=lambda item: (item[0], item[1]))
        tensors = [item[2] for item in ordered]
        target_file = save_dir / f"{exp_name}.npy"
        if tensors:
            np.save(target_file, np.stack(tensors))
        else:
            np.save(target_file, np.empty((0,)))
        print(f"Saved features for {exp_name} to {target_file}")

    meta_path = save_dir / "realtime_extract_config.json"
    with meta_path.open('w', encoding='utf-8') as f:
        json.dump(meta_info, f, indent=2, ensure_ascii=False)
    print(f"✅ 实时提取配置保存到: {meta_path}")

    return meta_info


def _prepare_layer_head_config(
    layer_head_config: Any,
    exp_names: Sequence[str],
    num_layers: int,
    num_heads: int,
) -> tuple[dict[str, dict[int, list[int]]], list[int]]:
    """
    将用户配置的层-头映射标准化为内部使用格式。
    """
    if layer_head_config is None:
        raise ValueError("layer_head_config must be provided for hidden state extraction.")

    def _resolve_config_for_exp(exp_name: str) -> Any:
        if callable(layer_head_config):
            return layer_head_config(exp_name)
        if isinstance(layer_head_config, dict):
            if exp_name in layer_head_config:
                return layer_head_config[exp_name]
            return layer_head_config.get("__default__")
        return layer_head_config

    normalized: dict[str, dict[int, list[int]]] = {}
    required_layers: set[int] = set()

    for exp_name in exp_names:
        raw_cfg = _resolve_config_for_exp(exp_name)
        if raw_cfg is None:
            raise ValueError(f"No layer/head configuration provided for experiment '{exp_name}'.")

        if isinstance(raw_cfg, Sequence) and not isinstance(raw_cfg, (str, bytes, dict)):
            raw_cfg = {int(layer_idx): "all" for layer_idx in raw_cfg}

        if not isinstance(raw_cfg, dict):
            raise TypeError(
                f"Layer/head configuration for experiment '{exp_name}' must be a dict "
                f"mapping layer indices to head indices (or 'all'). Got {type(raw_cfg)} instead."
            )

        layer_map: dict[int, list[int]] = {}
        for layer_idx_raw, raw_heads in raw_cfg.items():
            layer_idx = int(layer_idx_raw)
            if layer_idx < 0:
                layer_idx = num_layers + layer_idx
            if not 0 <= layer_idx < num_layers:
                raise ValueError(
                    f"Layer index {layer_idx_raw} (resolved to {layer_idx}) is out of range [0, {num_layers})."
                )

            if raw_heads is None or (isinstance(raw_heads, str) and raw_heads.lower() == "all"):
                head_indices = list(range(num_heads))
            else:
                if isinstance(raw_heads, Sequence) and not isinstance(raw_heads, (str, bytes)):
                    head_indices = []
                    for head_idx_raw in raw_heads:
                        head_idx = int(head_idx_raw)
                        if head_idx < 0:
                            head_idx = num_heads + head_idx
                        if not 0 <= head_idx < num_heads:
                            raise ValueError(
                                f"Head index {head_idx_raw} (resolved to {head_idx}) is out of range [0, {num_heads})."
                            )
                        head_indices.append(head_idx)
                    head_indices = sorted(set(head_indices))
                else:
                    raise TypeError(
                        f"Heads for layer {layer_idx} must be a sequence of indices or 'all'. "
                        f"Got {type(raw_heads)} instead."
                    )

            if not head_indices:
                continue
            layer_map[layer_idx] = head_indices
            required_layers.add(layer_idx)

        if not layer_map:
            raise ValueError(f"Experiment '{exp_name}' produced an empty layer/head configuration.")

        normalized[exp_name] = layer_map

    return normalized, sorted(required_layers)


def realtime_hidden_state_pre_extract(
    datas: dict[str, Any],
    exp_position_functions: dict[str, Callable[[list[str], list[str]], list[tuple[int, list[int]]]]],
    layer_head_config: Any,
    save_path: str,
    batchsize: int = 16,
    process_start_id: int = 0,
    input_key: str = "source",
    output_key: str = "target",
) -> dict[str, Any]:
    """
    实时提取指定token在特定层与head的hidden states。

    Args:
        datas: 经过load_input读取的原始数据。
        exp_position_functions: 位置提取函数，返回[(token_idx, _), ...]结构。
        layer_head_config: 指定提取的层/头，可为dict或可调用对象。
        save_path: 保存目录。
        batchsize: 批大小。
        process_start_id: 起始样本索引。
        input_key: 输入字段名。
        output_key: 输出字段名。

    Returns:
        dict: 元信息，包含提取配置。
    """
    global tokenizer, model
    if tokenizer is None or model is None:
        raise RuntimeError("Tokenizer or model is not loaded. Call load_model first.")
    model.eval()

    messages: list[Any] = get_message(datas)
    sources: list[str] = [d[input_key] for d in datas['metadata']['result']]
    targets: list[str] = [d[output_key] for d in datas['metadata']['result']]

    input_str0: str = tokenizer.apply_chat_template(messages[0], tokenize=False)
    instruct: str = input_str0[:input_str0.find(sources[0])]
    instruct_token_len: int = len(tokenizer.encode(instruct))
    print(f"instruct_token_len = {instruct_token_len}")

    src_tgt_tokens = [
        (_str2token_list(src), _str2token_list(tgt))
        for src, tgt in zip(sources, targets)
    ]
    exp_names = list(exp_position_functions.keys())
    exp_positions = [
        [
            exp_position_function(src_tokens, tgt_tokens)
            for src_tokens, tgt_tokens in src_tgt_tokens
        ]
        for exp_position_function in exp_position_functions.values()
    ]

    exp_hidden_states: dict[str, list[tuple[int, int, np.ndarray]]] = {
        exp_name: [] for exp_name in exp_names
    }

    total_samples = len(messages)
    normalized_layer_head_config: Optional[dict[str, dict[int, list[int]]]] = None
    required_layers: list[int] = []
    num_layers = 0
    num_heads = getattr(model.config, "num_attention_heads", None)
    head_dim_by_layer: dict[int, int] = {}

    for start_id in tqdm(range(process_start_id, total_samples, batchsize), desc="实时提取Hidden States"):
        end_id: int = min(start_id + batchsize, total_samples)
        batch_messages = messages[start_id:end_id]
        input_ids = tokenizer.apply_chat_template(
            batch_messages,
            padding=True,
            return_tensors="pt"
        ).to('cuda')

        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                output_attentions=False,
                output_hidden_states=True,
            )

        hidden_states = outputs.hidden_states
        if hidden_states is None:
            raise RuntimeError("Model did not return hidden states; ensure output_hidden_states=True.")

        if not num_layers:
            num_layers = len(hidden_states) - 1
            if num_heads is None:
                num_heads = getattr(model.config, "num_key_value_heads", None)
            if num_heads is None:
                raise RuntimeError("Cannot determine number of attention heads from model configuration.")
            normalized_layer_head_config, required_layers = _prepare_layer_head_config(
                layer_head_config, exp_names, num_layers, num_heads
            )

        layer_cache: dict[int, torch.Tensor] = {
            layer_idx: hidden_states[layer_idx + 1].detach().to(torch.float32).cpu()
            for layer_idx in required_layers
        }

        for layer_idx, tensor in layer_cache.items():
            head_dim_by_layer[layer_idx] = tensor.shape[-1] // num_heads

        for batch_offset, sample_idx in enumerate(range(start_id, end_id)):
            for eid, e_pos_list in enumerate(exp_positions):
                exp_name = exp_names[eid]
                token_config = normalized_layer_head_config[exp_name]
                for pos_idx, (qid, _kids) in enumerate(e_pos_list[sample_idx]):
                    if qid is None:
                        continue

                    per_layers: list[np.ndarray] = []
                    valid = True
                    for layer_idx, head_indices in sorted(token_config.items()):
                        layer_tensor = layer_cache[layer_idx][batch_offset]
                        if not (0 <= qid < layer_tensor.shape[0]):
                            valid = False
                            break
                        token_hidden = layer_tensor[qid].numpy()
                        head_dim = head_dim_by_layer[layer_idx]
                        pieces = []
                        for head_idx in head_indices:
                            start = head_idx * head_dim
                            end = start + head_dim
                            if end > token_hidden.shape[-1]:
                                valid = False
                                break
                            pieces.append(token_hidden[start:end])
                        if not valid:
                            break
                        per_layers.append(np.concatenate(pieces, axis=-1))
                    if not valid or not per_layers:
                        continue

                    concatenated = np.concatenate(per_layers, axis=-1)
                    exp_hidden_states[exp_name].append((sample_idx, pos_idx, concatenated))

        del layer_cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    save_dir = Path(save_path)
    save_dir.mkdir(parents=True, exist_ok=True)

    meta_info = {
        "instruct_token_len": instruct_token_len,
        "total_samples": total_samples,
        "num_layers": num_layers,
        "num_heads": num_heads,
        "head_dim_by_layer": head_dim_by_layer,
        "batchsize": batchsize,
        "save_path": str(save_dir),
        "layer_head_config": layer_head_config,
    }

    for exp_name in exp_names:
        ordered = sorted(exp_hidden_states[exp_name], key=lambda item: (item[0], item[1]))
        vectors = [item[2] for item in ordered]
        target_file = save_dir / f"{exp_name}_hidden.npy"
        if vectors:
            np.save(target_file, np.stack(vectors))
        else:
            np.save(target_file, np.empty((0,)))
        print(f"Saved hidden states for {exp_name} to {target_file}")

    meta_path = save_dir / "realtime_hidden_config.json"
    with meta_path.open('w', encoding='utf-8') as f:
        json.dump(meta_info, f, indent=2, ensure_ascii=False)
    print(f"✅ Hidden state提取配置保存到: {meta_path}")

    return meta_info

# 可选：添加数据加载函数
def load_attention_data(save_path, batch_id=None):
    """
    加载保存的attention数据
    
    Args:
        save_path: 保存路径
        batch_id: 如果指定，只加载特定batch；否则加载完整数据
    
    Returns:
        dict: 包含attention数据和元数据的字典
    """
    if batch_id is not None:
        # 加载特定batch
        files = [f for f in os.listdir(save_path) if f.startswith(f'attention_batch_{batch_id}')]
        if not files:
            raise FileNotFoundError(f"未找到batch {batch_id}的数据")
        file_path = os.path.join(save_path, files[0])
    else:
        # 加载完整数据
        file_path = os.path.join(save_path, 'all_attentions.npz')
    
    data = np.load(file_path, allow_pickle=True)
    
    # 加载配置
    config_path = os.path.join(save_path, 'config.json')
    if os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
    else:
        config = {}
    
    return {
        'attentions': data['attentions'],
        'sources': data['sources'],
        'predictions': data['predictions'],
        'config': config
    }


def config_realtime_pre_extract():
    # Qwen2.5-VL-7B-Instruct_cscd-ns-test_step
    # init_config = {
    #     "model_path": "/data/images/llms/Qwen/Qwen2.5-VL-7B-Instruct",
    #     "device_ids": "6",
    #     "message_path": "/home/yangchunhao/csc/results/Qwen2.5-VL-7B-Instruct_cscd-ns-test_step/checkpoints/csc.json",
    #     "max_len": 10,
    #     "batchsize": 6,
    #     "process_start_id": 0,
    #     "save_path": "/home/yangchunhao/csc/exp/p2p/cscd-ns/test_debug"
    # }
    # exp_position_functions = {
    #     "normal_token": get_normal_token_coresponding,
    #     "first_diff_token": find_first_diff
    # }

    # init_config = {
    #     "model_path": "/data/images/llms/Qwen/Qwen2.5-VL-7B-Instruct",
    #     "device_ids": "5",
    #     "message_path": "/home/yangchunhao/csc/results/Qwen2.5-VL-7B-Instruct_cscd-ns-train_step/checkpoints/csc.json",
    #     "max_len": None,
    #     "batchsize": 4,
    #     "process_start_id": 0,
    #     "save_path": "/home/yangchunhao/csc/exp/p2p/cscd-ns/train_realtime"
    # }
    # exp_position_functions = {
    #     "normal_token": get_normal_token_coresponding,
    #     "first_diff_token": find_first_diff,
    # }

    # Qwen2.5-VL-7B-Instruct_cscd-ns-test_step with instruction and more fetures
    # init_config = {
    #     "model_path": "/data/images/llms/Qwen/Qwen2.5-VL-7B-Instruct",
    #     "device_ids": "6",
    #     "message_path": "/home/yangchunhao/csc/results/Qwen2.5-VL-7B-Instruct_cscd-ns-test_step/checkpoints/csc.json",
    #     "max_len": None,
    #     "batchsize": 6,
    #     "process_start_id": 0,
    #     "save_path": "/home/yangchunhao/csc/exp/p2p/cscd-ns/test_add_instruction",
    #     "add_instruction": True,
    #     "crop_shift": 5,
    # }
    # exp_position_functions = {
    #     "first_diff_token": partial(first_diff_p2minus_k, k=0),
    #     "normal_token": partial(normal_token_p2minus_k, k=0),
    #     "first_diff_p2minus3": partial(first_diff_p2minus_k, k=3),
    #     "normal_p2minus3": partial(normal_token_p2minus_k, k=3),
    #     "first_diff_p2minus4": partial(first_diff_p2minus_k, k=4),
    #     "normal_p2minus4": partial(normal_token_p2minus_k, k=4),
    # }

    # ----------------------more future config----------------------

    # Qwen2.5-VL-7B-Instruct_cscd-ns-dev_step with instruction and more fetures
    # init_config = {
    #     "model_path": "/data/images/llms/Qwen/Qwen2.5-VL-7B-Instruct",
    #     "device_ids": "6",
    #     "message_path": "/home/yangchunhao/csc/results/Qwen2.5-VL-7B-Instruct_cscd-ns-dev_step/checkpoints/csc.json",
    #     "max_len": None,
    #     "batchsize": 6,
    #     "process_start_id": 0,
    #     "save_path": "/home/yangchunhao/csc/exp/p2p/cscd-ns/dev_add_instruction",
    #     "add_instruction": True,
    #     "crop_shift": 5,
    # }
    # exp_position_functions = {
    #     "first_diff_token": partial(first_diff_p2minus_k, k=0),
    #     "normal_token": partial(normal_token_p2minus_k, k=0),
    #     "first_diff_p2minus3": partial(first_diff_p2minus_k, k=3),
    #     "normal_p2minus3": partial(normal_token_p2minus_k, k=3),
    #     "first_diff_p2minus4": partial(first_diff_p2minus_k, k=4),
    #     "normal_p2minus4": partial(normal_token_p2minus_k, k=4),
    # }

    # Qwen2.5-VL-7B-Instruct_cscd-ns-train_step
    # init_config = {
    #     "model_path": "/data/images/llms/Qwen/Qwen2.5-VL-7B-Instruct",
    #     "device_ids": "3",
    #     "message_path": "/home/yangchunhao/csc/results/Qwen2.5-VL-7B-Instruct_cscd-ns-train_step/checkpoints/csc.json",
    #     "max_len": None,
    #     "batchsize": 6,
    #     "process_start_id": 0,
    #     "save_path": "/home/yangchunhao/csc/exp/p2p/cscd-ns/train_add_instruction",
    #     "add_instruction": True,
    #     "crop_shift": 5,
    # }
    # exp_position_functions = {
    #     "first_diff_token": partial(first_diff_p2minus_k, k=0),
    #     "normal_token": partial(normal_token_p2minus_k, k=0),
    #     "first_diff_p2minus3": partial(first_diff_p2minus_k, k=3),
    #     "normal_p2minus3": partial(normal_token_p2minus_k, k=3),
    #     "first_diff_p2minus4": partial(first_diff_p2minus_k, k=4),
    #     "normal_p2minus4": partial(normal_token_p2minus_k, k=4),
    # }


    # ----------------------copy config----------------------
    
    # init_config = {
    #     "model_path": "/data/images/llms/Qwen/Qwen2.5-VL-7B-Instruct",
    #     "device_ids": "6",
    #     "message_path": "/home/yangchunhao/csc/results/Qwen2.5-VL-7B-Instruct_cscd-ns-test_copy/checkpoints/csc.json",
    #     "max_len": None,
    #     "batchsize": 6,
    #     "process_start_id": 0,
    #     "save_path": "/home/yangchunhao/csc/exp/p2p/cscd-ns/test_add_instruction",
    #     "add_instruction": True,
    #     "crop_shift": 5,
    #     "input_key": "source",
    #     "output_key": "source"
    # }
    # exp_position_functions = {
    #     "first_diff_token": partial(first_diff_p2minus_k, k=0),
    #     "normal_token": partial(normal_token_p2minus_k, k=0),
    #     "first_diff_p2minus3": partial(first_diff_p2minus_k, k=3),
    #     "normal_p2minus3": partial(normal_token_p2minus_k, k=3),
    #     "first_diff_p2minus4": partial(first_diff_p2minus_k, k=4),
    #     "normal_p2minus4": partial(normal_token_p2minus_k, k=4),
    # }
    

    # # ----------------------src2src tgt2tgt config----------------------
    # init_config = {
    #     "model_path": "/data/images/llms/Qwen/Qwen2.5-VL-7B-Instruct",
    #     "device_ids": "6",
    #     "message_path": "/home/yangchunhao/csc/results/Qwen2.5-VL-7B-Instruct_cscd-ns-test_step/checkpoints/csc.json",
    #     "max_len": None,
    #     "batchsize": 6,
    #     "process_start_id": 0,
    #     "save_path": "/home/yangchunhao/csc/exp/p2p/cscd-ns/test",
    #     "add_instruction": True,
    #     "crop_shift": 5,
    #     "input_key": "source",
    #     "output_key": "target",
    # }
    # exp_position_functions = {
    #     "first_diff_token": partial(first_diff_p2minus_k, k=0),
    #     "normal_token": partial(normal_token_p2minus_k, k=0),
    #     # "first_diff_p2minus3": partial(first_diff_p2minus_k, k=3),
    #     # "normal_p2minus3": partial(normal_token_p2minus_k, k=3),
    #     # "first_diff_p2minus4": partial(first_diff_p2minus_k, k=4),
    #     # "normal_p2minus4": partial(normal_token_p2minus_k, k=4),
    #     "first_diff_p2minue1_t2t": partial(first_diff_p2minus_k, k=1, to_target=True),
    #     "normal_token_p2minue1_t2t": partial(normal_token_p2minus_k, k=1, to_target=True),
    #     "first_diff_p2minue1_s2s": partial(first_diff_p2minus_k, k=1, from_target=False),
    #     "normal_token_p2minue1_s2s": partial(normal_token_p2minus_k, k=1, from_target=False),
    #     "first_diff_p2minue2_t2t": partial(first_diff_p2minus_k, k=2, to_target=True),
    #     "normal_token_p2minue2_t2t": partial(normal_token_p2minus_k, k=2, to_target=True),
    #     "first_diff_p2minue2_s2s": partial(first_diff_p2minus_k, k=2, from_target=False),
    #     "normal_token_p2minue2_s2s": partial(normal_token_p2minus_k, k=2, from_target=False),
    #     "first_diff_p2minue3_t2t": partial(first_diff_p2minus_k, k=3, to_target=True),
    #     "normal_token_p2minue3_t2t": partial(normal_token_p2minus_k, k=3, to_target=True),
    #     "first_diff_p2minue3_s2s": partial(first_diff_p2minus_k, k=3, from_target=False),
    #     "normal_token_p2minue3_s2s": partial(normal_token_p2minus_k, k=3, from_target=False),
    # }
    
    init_config = {
        "model_path": "/data/images/llms/Qwen/Qwen2.5-VL-7B-Instruct",
        "device_ids": "6",
        "message_path": "/home/yangchunhao/csc/results/Qwen2.5-VL-7B-Instruct_cscd-ns-train_step/checkpoints/csc.json",
        "max_len": None,
        "batchsize": 6,
        "process_start_id": 0,
        "save_path": "/home/yangchunhao/csc/exp/p2p/cscd-ns/train",
        "add_instruction": True,
        "crop_shift": 5,
        "input_key": "source",
        "output_key": "target",
    }
    exp_position_functions = {
        "first_diff_token": partial(first_diff_p2minus_k, k=0),
        "normal_token": partial(normal_token_p2minus_k, k=0),
        # "first_diff_p2minus3": partial(first_diff_p2minus_k, k=3),
        # "normal_p2minus3": partial(normal_token_p2minus_k, k=3),
        # "first_diff_p2minus4": partial(first_diff_p2minus_k, k=4),
        # "normal_p2minus4": partial(normal_token_p2minus_k, k=4),
        "first_diff_p2minue1_t2t": partial(first_diff_p2minus_k, k=1, to_target=True),
        "normal_token_p2minue1_t2t": partial(normal_token_p2minus_k, k=1, to_target=True),
        "first_diff_p2minue1_s2s": partial(first_diff_p2minus_k, k=1, from_target=False),
        "normal_token_p2minue1_s2s": partial(normal_token_p2minus_k, k=1, from_target=False),
        "first_diff_p2minue2_t2t": partial(first_diff_p2minus_k, k=2, to_target=True),
        "normal_token_p2minue2_t2t": partial(normal_token_p2minus_k, k=2, to_target=True),
        "first_diff_p2minue2_s2s": partial(first_diff_p2minus_k, k=2, from_target=False),
        "normal_token_p2minue2_s2s": partial(normal_token_p2minus_k, k=2, from_target=False),
        "first_diff_p2minue3_t2t": partial(first_diff_p2minus_k, k=3, to_target=True),
        "normal_token_p2minue3_t2t": partial(normal_token_p2minus_k, k=3, to_target=True),
        "first_diff_p2minue3_s2s": partial(first_diff_p2minus_k, k=3, from_target=False),
        "normal_token_p2minue3_s2s": partial(normal_token_p2minus_k, k=3, from_target=False),
    }
    
    # ----------------------debug config----------------------
    # init_config = {
    #     "model_path": "/data/images/llms/Qwen/Qwen2.5-VL-7B-Instruct",
    #     "device_ids": "6",
    #     "message_path": "/home/yangchunhao/csc/results/Qwen2.5-VL-7B-Instruct_cscd-ns-test_step/checkpoints/csc.json",
    #     "max_len": 10,
    #     "batchsize": 8,
    #     "process_start_id": 0,
    #     "save_path": "/home/yangchunhao/csc/exp/p2p/cscd-ns/test_debug",
    #     "add_instruction": True,
    #     "crop_shift": 5,
    # }
    # exp_position_functions = {
    #     "first_diff_token": partial(first_diff_p2minus_k, k=0),
    #     "normal_token": partial(normal_token_p2minus_k, k=0),
    #     "first_diff_p2minus3": partial(first_diff_p2minus_k, k=2),
    #     "normal_p2minus3": partial(normal_token_p2minus_k, k=4),
    #     "first_diff_p2minus4": partial(first_diff_p2minus_k, k=4),
    #     "normal_p2minus4": partial(normal_token_p2minus_k, k=4),
    # }
    
    # init_config = {
    #     "model_path": "/data/images/llms/Qwen/Qwen2.5-VL-7B-Instruct",
    #     "device_ids": "6",
    #     "message_path": "/home/yangchunhao/csc/results/Qwen2.5-VL-7B-Instruct_cscd-ns-test_step/checkpoints/csc.json",
    #     "max_len": None,
    #     "batchsize": 6,
    #     "process_start_id": 0,
    #     "save_path": "/home/yangchunhao/csc/exp/p2p/cscd-ns/test",
    #     "add_instruction": True,
    #     "crop_shift": 5,
    #     "input_key": "source",
    #     "output_key": "target",
    # }
    # exp_position_functions = {
    #     "first_diff_token": partial(first_diff_p2minus_k, k=0),
    #     "normal_token": partial(normal_token_p2minus_k, k=0),
    #     # "first_diff_p2minus3": partial(first_diff_p2minus_k, k=3),
    #     # "normal_p2minus3": partial(normal_token_p2minus_k, k=3),
    #     # "first_diff_p2minus4": partial(first_diff_p2minus_k, k=4),
    #     # "normal_p2minus4": partial(normal_token_p2minus_k, k=4),
    #     "first_diff_p2minue1_t2t": partial(first_diff_p2minus_k, k=1, to_target=True),
    #     "normal_token_p2minue1_t2t": partial(normal_token_p2minus_k, k=1, to_target=True),
    #     "first_diff_p2minue1_s2s": partial(first_diff_p2minus_k, k=1, from_target=False),
    #     "normal_token_p2minue1_s2s": partial(normal_token_p2minus_k, k=1, from_target=False),
    #     "first_diff_p2minue2_t2t": partial(first_diff_p2minus_k, k=2, to_target=True),
    #     "normal_token_p2minue2_t2t": partial(normal_token_p2minus_k, k=2, to_target=True),
    #     "first_diff_p2minue2_s2s": partial(first_diff_p2minus_k, k=2, from_target=False),
    #     "normal_token_p2minue2_s2s": partial(normal_token_p2minus_k, k=2, from_target=False),
    #     "first_diff_p2minue3_t2t": partial(first_diff_p2minus_k, k=3, to_target=True),
    #     "normal_token_p2minue3_t2t": partial(normal_token_p2minus_k, k=3, to_target=True),
    #     "first_diff_p2minue3_s2s": partial(first_diff_p2minus_k, k=3, from_target=False),
    #     "normal_token_p2minue3_s2s": partial(normal_token_p2minus_k, k=3, from_target=False),
    # }
    return init_config, exp_position_functions


def config_realtime_hidden_state_pre_extract():
    """
    示例：抽取target中first_diff_token在第14层（索引13）所有head的hidden states。
    """
    init_config = {
        "model_path": "/data/images/llms/Qwen/Qwen2.5-VL-7B-Instruct",
        "device_ids": "6",
        "message_path": "/home/yangchunhao/csc/results/Qwen2.5-VL-7B-Instruct_cscd-ns-test_step/checkpoints/csc.json",
        "max_len": 100,
        "batchsize": 6,
        "process_start_id": 0,
        "save_path": "/home/yangchunhao/csc/exp/hidden_states/cscd-ns/test",
        "input_key": "source",
        "output_key": "target",
    }
    exp_position_functions = {
        "first_diff_token": find_first_diff,
    }
    layer_head_config = {
        "first_diff_token": {
            13: "all",  # 第14层（索引13），提取所有head
        }
    }
    return init_config, exp_position_functions, layer_head_config


if __name__ == "__main__":
    # 谨慎调整batchsize，批处理数据不同会在某些位置上明显影响效果。
    # python -m exp.statistic_attn
    # init_config, exp_position_functions = config_realtime_pre_extract()
    # load_model(model_path = init_config['model_path'], device_ids = init_config['device_ids'])
    # datas = load_input(init_config['message_path'], max_len=init_config['max_len'])
    
    # statistic_attn(
    #     datas,
    #     save_path = "/home/yangchunhao/csc/exp/attn/cscd-ns_test_vl_copy",
    #     batchsize=6,
    #     input_key=init_config['input_key'],
    #     output_key=init_config['output_key'],
    # )
    
    # realtime_pre_extract(
    #     datas,
    #     exp_position_functions,
    #     save_path=init_config['save_path'],
    #     batchsize=init_config['batchsize'],
    #     process_start_id=init_config['process_start_id'],
    #     add_instruction=init_config['add_instruction'],
    #     crop_shift=init_config['crop_shift'],
    #     input_key=init_config['input_key'],
    #     output_key=init_config['output_key'],
    # )


    # # 如需抽取hidden states，可参考以下示例：
    init_config, exp_position_functions, layer_head_config = config_realtime_hidden_state_pre_extract()
    load_model(model_path=init_config['model_path'], device_ids=init_config['device_ids'])
    datas = load_input(init_config['message_path'], max_len=init_config['max_len'])
    realtime_hidden_state_pre_extract(
        datas,
        exp_position_functions,
        layer_head_config=layer_head_config,
        save_path=init_config['save_path'],
        batchsize=init_config['batchsize'],
        process_start_id=init_config['process_start_id'],
        input_key=init_config['input_key'],
        output_key=init_config['output_key'],
    )
