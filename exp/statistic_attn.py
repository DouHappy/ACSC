from optparse import Option
import os
import json
from tqdm import tqdm
from typing import Any, Optional
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

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
    
    if start_id:
        datas['metadata']['result'] = datas['metadata']['result'][start_id:]
    if max_len:
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
    import json
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

if __name__ == "__main__":
    load_model(model_path = "/data/images/llms/Qwen/Qwen2.5-VL-7B-Instruct", device_ids = "6")
    datas = load_input(message_path= "/home/yangchunhao/csc/results/Qwen2.5-VL-7B-Instruct_cscd-ns-test_step/checkpoints/csc.json", max_len=5000)
    
    statistic_attn(datas, save_path = "/home/yangchunhao/csc/exp/attn/cscd-ns_test_vl_step/src_tgt", batchsize=6)