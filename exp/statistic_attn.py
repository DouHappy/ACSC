import os
import json
from tqdm import tqdm
from typing import Any, Optional
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from pathlib import Path

from exp.analyzer import get_normal_token_coresponding, find_first_diff

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
        device_map='auto', 
        torch_dtype=torch.bfloat16,
        ).to("cuda")

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
                "content": d['prediction'],
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

def statistic_attn(datas, save_path, batchsize=16, process_start_id=0):
    # 获取message
    global tokenizer, model
    messages: list[Any] = get_message(datas)
    # 获取基本的输入
    sources: list[str] = [d['source'] for d in datas['metadata']['result']]
    predictions: list[str] = [d['target'] for d in datas['metadata']['result']]
    
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
) -> dict[str, Any]:
    """
    实时推理attention并提取特征，避免中间磁盘I/O。
    """
    global tokenizer, model
    if tokenizer is None or model is None:
        raise RuntimeError("Tokenizer or model is not loaded. Call load_model first.")
    model.eval()

    messages: list[Any] = get_message(datas)
    sources: list[str] = [d['source'] for d in datas['metadata']['result']]
    targets: list[str] = [d['target'] for d in datas['metadata']['result']]

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
        for layer_attn in attentions:
            cropped = layer_attn[:, :, instruct_token_len:, instruct_token_len:]
            cropped_layers.append(cropped.to(torch.float32).cpu())
        batch_attentions = torch.stack(cropped_layers, dim=0).numpy()

        for batch_offset, sample_idx in enumerate(range(start_id, end_id)):
            attn_matrix = batch_attentions[:, batch_offset]
            for eid, e_pos_list in enumerate(exp_positions):
                for pos_idx, (qid, kids) in enumerate(e_pos_list[sample_idx]):
                    if qid is None:
                        continue
                    exp_attn_matrix[exp_names[eid]].append(
                        (sample_idx, pos_idx, attn_matrix[:, :, qid, kids])
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
    init_config = {
        "model_path": "/data/images/llms/Qwen/Qwen2.5-VL-7B-Instruct",
        "device_ids": "6",
        "message_path": "/home/yangchunhao/csc/results/Qwen2.5-VL-7B-Instruct_cscd-ns-train_step/checkpoints/csc.json",
        "max_len": 5000,
        "batchsize": 6,
        "process_start_id": 0,
        "save_path": "/home/yangchunhao/csc/exp/p2p/cscd-ns/train_realtime"
    }
    exp_position_functions = {
        "normal_token": get_normal_token_coresponding,
        "first_diff_token": find_first_diff,
    }
    return init_config, exp_position_functions


if __name__ == "__main__":
    load_model(model_path = "/data/images/llms/Qwen/Qwen2.5-VL-7B-Instruct", device_ids = "6")
    datas = load_input(message_path= "/home/yangchunhao/csc/results/Qwen2.5-VL-7B-Instruct_cscd-ns-test_step/checkpoints/csc.json", max_len=5000)
    
    # statistic_attn(datas, save_path = "/home/yangchunhao/csc/exp/attn/cscd-ns_test_vl_step/src_tgt", batchsize=6)
    init_config, exp_position_functions = config_realtime_pre_extract()
    realtime_pre_extract(
        datas,
        exp_position_functions,
        save_path=init_config['save_path'],
        batchsize=init_config['batchsize'],
        process_start_id=init_config['process_start_id'],
    )