import torch
import numpy as np
import os
import json
from tqdm import tqdm
from typing import List, Dict, Tuple, Optional, Union, Any
import argparse
from dataclasses import dataclass
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("/data/images/llms/Qwen/Qwen2.5-VL-7B-Instruct")
class AttentionAnalyzer:
    """高效处理注意力数据，统计8类token的注意力分布"""
    
    # 定义8种位置类型
    POSITION_TYPES = [
        "src_normal", 
        "src_edited_left", "src_edited_mid", "src_edited_right",
        "prdt_normal", 
        "prdt_edited_left", "prdt_edited_mid", "prdt_edited_right"
    ]
    
    def __init__(self, attn_dir: str, sources: List[str], predictions: List[str]):
        """
        初始化注意力分析器
        
        Args:
            attn_dir: 注意力数据目录
            sources: 原始文本列表
            predictions: 预测文本列表
        """
        self.attn_dir = attn_dir
        self.sources = sources
        self.predictions = predictions
        
        # 初始化全局统计量
        self._reset_statistics()
    
    def _reset_statistics(self):
        """重置所有统计量"""
        # 初始化全局统计量 (使用float64避免累加精度问题)
        self.src_normal_token_st = {"max": None, "avg": None}
        self.src_edited_token_st = {"max": [None, None, None], "avg": [None, None, None]}
        self.prdt_normal_token_st = {"max": None, "avg": None}
        self.prdt_edited_token_st = {"max": [None, None, None], "avg": [None, None, None]}
        
        # 初始化计数器
        self.num_src_normal_token = 0
        self.num_src_edited_token = [0, 0, 0]  # [left, mid, right]
        self.num_prdt_normal_token = 0
        self.num_prdt_edited_token = [0, 0, 0]  # [left, mid, right]
    
    def analyze(self) -> Dict[str, Any]:
        """
        执行注意力分析
        
        Returns:
            统计结果字典
        """
        self._reset_statistics()
        
        # 遍历所有注意力文件
        for file in tqdm(self._load_attentions(), desc="Processing attention files"):
            self._process_attention_file(file)
        
        return self._calculate_results()
    
    def _load_attentions(self) -> List[str]:
        """加载注意力文件列表"""
        # 这里假设存在一个全局函数load_attentions
        # 实际使用时可能需要替换为实际实现
        return [f for f in os.listdir(self.attn_dir) if f.endswith('.npz')]
    
    def _process_attention_file(self, file: str):
        """处理单个注意力文件"""
        attn_path = os.path.join(self.attn_dir, file)
        attn_metadata = np.load(attn_path, allow_pickle=True)
        start_id = attn_metadata['start_id']
        end_id = attn_metadata['end_id']
        batch_size = end_id - start_id
        
        # 转换为PyTorch张量 [layer, batch, head, seq_len, seq_len]
        attentions = torch.from_numpy(attn_metadata['attentions']).to('cuda')
        L = attentions.size(-1)  # 序列总长度
        
        # 首次遇到时初始化统计量
        if self.src_normal_token_st["max"] is None:
            self._initialize_statistics(attentions)
        
        # 预计算注意力统计量 [layer, batch, head, L]
        attn_max = attentions.max(dim=-1)[0]  # 沿目标序列维度取最大
        attn_avg = attentions.mean(dim=-1)     # 沿目标序列维度取平均
        
        # 批量获取当前批次的文本和编辑操作
        batch_sources = [self._get_str_token(self.sources[start_id + i]) 
                         for i in range(batch_size)]
        batch_predictions = [self._get_str_token(self.predictions[start_id + i]) 
                             for i in range(batch_size)]
        batch_editions = [
            self._get_edit_operations(batch_sources[i], batch_predictions[i]) 
            for i in range(batch_size)
        ]
        
        # 预计算所有掩码 (batch_size x 8 x L)
        masks = self._build_batch_masks(
            batch_sources, batch_predictions, batch_editions, L
        )
        
        # 向量化累加统计量
        self._accumulate_statistics(attn_max, attn_avg, masks)
    
    def _initialize_statistics(self, attentions: torch.Tensor):
        """根据注意力张量初始化统计量"""
        layer, _, head = attentions.size(0), attentions.size(1), attentions.size(2)
        zeros = lambda: torch.zeros(layer, head, dtype=torch.float64).to('cuda')
        
        self.src_normal_token_st = {"max": zeros(), "avg": zeros()}
        self.src_edited_token_st = {"max": [zeros(), zeros(), zeros()], 
                                  "avg": [zeros(), zeros(), zeros()]}
        self.prdt_normal_token_st = {"max": zeros(), "avg": zeros()}
        self.prdt_edited_token_st = {"max": [zeros(), zeros(), zeros()], 
                                    "avg": [zeros(), zeros(), zeros()]}
    
    def _accumulate_statistics(
        self,
        attn_max: torch.Tensor,
        attn_avg: torch.Tensor,
        masks: List[torch.Tensor]
    ):
        """累加统计量到全局结果"""
        for op, attn_val in [("max", attn_max), ("avg", attn_avg)]:
            for mask_idx, mask_name in enumerate(self.POSITION_TYPES):
                mask_batch = masks[mask_idx]  # [batch_size, L]
                counts = mask_batch.sum(dim=1).to(torch.int64)  # [batch_size]
                
                # 向量化累加: [layer, batch, head, L] * [batch, L] -> [layer, batch, head]
                weighted_sum = (attn_val * mask_batch.unsqueeze(0).unsqueeze(2)).sum(dim=-1)
                
                # 累加到全局统计量
                if mask_name == "src_normal":
                    self.src_normal_token_st[op] += weighted_sum.sum(dim=1)
                    self.num_src_normal_token += counts.sum().item()
                elif mask_name.startswith("src_edited"):
                    pos = ["left", "mid", "right"].index(mask_name.split("_")[-1])
                    self.src_edited_token_st[op][pos] += weighted_sum.sum(dim=1)
                    self.num_src_edited_token[pos] += counts.sum().item()
                elif mask_name == "prdt_normal":
                    self.prdt_normal_token_st[op] += weighted_sum.sum(dim=1)
                    self.num_prdt_normal_token += counts.sum().item()
                elif mask_name.startswith("prdt_edited"):
                    pos = ["left", "mid", "right"].index(mask_name.split("_")[-1])
                    self.prdt_edited_token_st[op][pos] += weighted_sum.sum(dim=1)
                    self.num_prdt_edited_token[pos] += counts.sum().item()
    
    def _build_batch_masks(
        self,
        sources: List[List[str]],
        predictions: List[List[str]],
        editions_list: List[List[Tuple[int, int]]],
        L: int
    ) -> List[torch.Tensor]:
        """
        为整个批次构建8种位置类型的掩码
        
        Args:
            sources: 当前批次的source token列表
            predictions: 当前批次的prediction token列表
            editions_list: 当前批次的编辑操作列表
            L: 序列总长度
            
        Returns:
            masks: 8个掩码张量组成的列表 [batch_size, L]
        """
        batch_size = len(sources)
        device = 'cuda'  # 使用CPU处理（注意力数据通常较大）
        
        # 初始化8种掩码 [8, batch_size, L]
        masks = torch.zeros(8, batch_size, L, dtype=torch.bool, device=device)
        
        for i in range(batch_size):
            source_len = len(sources[i])
            pred_len = len(predictions[i])
            shift = source_len + 6  # prediction起始位置
            
            # 提取编辑位置
            source_edits = set(e[0] for e in editions_list[i])
            pred_edits = set(e[1] for e in editions_list[i])
            
            # 1. source normal: source部分非编辑位置
            masks[0, i, :source_len] = True
            for pos in source_edits:
                if 0 <= pos < source_len:
                    masks[0, i, pos] = False  # 排除编辑位置
            
            # 2. source edited left: 编辑位置的前一个
            for pos in source_edits:
                if 1 <= pos < source_len:
                    masks[1, i, pos-1] = True
            
            # 3. source edited mid: 编辑位置本身
            for pos in source_edits:
                if 0 <= pos < source_len:
                    masks[2, i, pos] = True
            
            # 4. source edited right: 编辑位置的后一个
            for pos in source_edits:
                if 0 <= pos < source_len-1:
                    masks[3, i, pos+1] = True
            
            # 5. prediction normal: prediction部分非编辑位置
            masks[4, i, shift:shift+pred_len] = True
            for pos in pred_edits:
                if 0 <= pos < pred_len:
                    masks[4, i, shift+pos] = False  # 排除编辑位置
            
            # 6. prediction edited left: 编辑位置的前一个
            for pos in pred_edits:
                if 1 <= pos < pred_len:
                    masks[5, i, shift+pos-1] = True
            
            # 7. prediction edited mid: 编辑位置本身
            for pos in pred_edits:
                if 0 <= pos < pred_len:
                    masks[6, i, shift+pos] = True
            
            # 8. prediction edited right: 编辑位置的后一个
            for pos in pred_edits:
                if 0 <= pos < pred_len-1:
                    masks[7, i, shift+pos+1] = True
        
        # 修正source normal: 排除所有编辑相关位置
        edit_masks = masks[1:4].any(dim=0)  # [batch_size, L]
        masks[0] &= ~edit_masks
        
        # 修正prediction normal: 排除所有编辑相关位置
        edit_masks = masks[5:8].any(dim=0)  # [batch_size, L]
        masks[4] &= ~edit_masks
        
        return [masks[i] for i in range(8)]  # 返回8个独立的掩码张量
    
    def _calculate_results(self) -> Dict[str, Any]:
        """计算最终统计结果"""
        return {
            "src_normal": {
                "max": self._safe_divide(self.src_normal_token_st["max"], self.num_src_normal_token),
                "avg": self._safe_divide(self.src_normal_token_st["avg"], self.num_src_normal_token)
            },
            "src_edited": {
                "left": {
                    "max": self._safe_divide(self.src_edited_token_st["max"][0], self.num_src_edited_token[0]),
                    "avg": self._safe_divide(self.src_edited_token_st["avg"][0], self.num_src_edited_token[0])
                },
                "mid": {
                    "max": self._safe_divide(self.src_edited_token_st["max"][1], self.num_src_edited_token[1]),
                    "avg": self._safe_divide(self.src_edited_token_st["avg"][1], self.num_src_edited_token[1])
                },
                "right": {
                    "max": self._safe_divide(self.src_edited_token_st["max"][2], self.num_src_edited_token[2]),
                    "avg": self._safe_divide(self.src_edited_token_st["avg"][2], self.num_src_edited_token[2])
                }
            },
            "prdt_normal": {
                "max": self._safe_divide(self.prdt_normal_token_st["max"], self.num_prdt_normal_token),
                "avg": self._safe_divide(self.prdt_normal_token_st["avg"], self.num_prdt_normal_token)
            },
            "prdt_edited": {
                "left": {
                    "max": self._safe_divide(self.prdt_edited_token_st["max"][0], self.num_prdt_edited_token[0]),
                    "avg": self._safe_divide(self.prdt_edited_token_st["avg"][0], self.num_prdt_edited_token[0])
                },
                "mid": {
                    "max": self._safe_divide(self.prdt_edited_token_st["max"][1], self.num_prdt_edited_token[1]),
                    "avg": self._safe_divide(self.prdt_edited_token_st["avg"][1], self.num_prdt_edited_token[1])
                },
                "right": {
                    "max": self._safe_divide(self.prdt_edited_token_st["max"][2], self.num_prdt_edited_token[2]),
                    "avg": self._safe_divide(self.prdt_edited_token_st["avg"][2], self.num_prdt_edited_token[2])
                }
            }
        }
    
    def _safe_divide(
        self, 
        tensor: Optional[torch.Tensor], 
        denominator: int
    ) -> Optional[torch.Tensor]:
        """安全除法，处理分母为0的情况"""
        if denominator == 0 or tensor is None:
            return None
        return tensor / denominator
    
    # ===========================================
    # 以下方法需要用户根据实际情况实现
    # ===========================================
    def _get_str_token(self, text: str) -> List[str]:
        """将文本转换为token列表（需用户实现）"""
        return [
            tokenizer.decode(t) for t in tokenizer.encode(text)
        ]
    
    def _get_edit_operations(
        self, 
        src_tokens: List[str], 
        tgt_tokens: List[str]
    ) -> List[Tuple[int, int]]:
        """获取编辑操作列表（需用户实现）"""
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

class TargetGenerater:
    def __init__(self, tokenizer_path, data_path):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.data_path = data_path
        self.datas = load_json(self.data_path)
        
    def get_target(self):
        sources: list[str] = [d['source'] for d in self.datas['metadata']['result']]
        predictions: list[str] = [d['target'] for d in self.datas['metadata']['result']]
        return sources, predictions

    def get_str_token(self, input):
        return [
            self.tokenizer.decode(t) for t in self.tokenizer.encode(input)
        ]

"""
draw
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, Optional, Tuple, List

def plot_attention_results(
    results: Dict[str, Any],
    operation: str = "max",
    fig_save_path: str = "./attention_plots",
    output_npz: bool = True,
    output_png: bool = True,
    dpi: int = 300
) -> None:
    """
    可视化注意力分析结果
    
    Args:
        results: AttentionAnalyzer.analyze() 返回的结果字典
        operation: 要可视化的操作类型 ("max" 或 "avg")
        fig_save_path: 图片保存路径
        output_npz: 是否保存npz数据文件
        output_png: 是否保存PNG图片
        dpi: 图片分辨率
    """
    # 创建保存目录
    os.makedirs(fig_save_path, exist_ok=True)
    
    # 验证操作类型
    if operation not in ["max", "avg"]:
        raise ValueError(f"Invalid operation type: {operation}. Must be 'max' or 'avg'")
    
    # 提取需要可视化的数据
    datas = []
    titles = [
        "Source Normal", 
        "Source Edited (p-1)", "Source Edited (p)", "Source Edited (p+1)",
        "Prediction Normal", 
        "Prediction Edited (p-1)", "Prediction Edited (p)", "Prediction Edited (p+1)"
    ]
    
    # 1. Source Normal
    src_normal = results["src_normal"][operation]
    datas.append(src_normal if src_normal is not None else None)
    
    # 2-4. Source Edited (left, mid, right)
    for pos in ["left", "mid", "right"]:
        src_edited = results["src_edited"][pos][operation]
        if src_edited is not None and src_normal is not None:
            # 计算相对值（相对于正常token）
            ratio = src_edited / src_normal
            datas.append(ratio)
        else:
            datas.append(None)
    
    # 5. Prediction Normal
    prdt_normal = results["prdt_normal"][operation]
    datas.append(prdt_normal if prdt_normal is not None else None)
    
    # 6-8. Prediction Edited (left, mid, right)
    for pos in ["left", "mid", "right"]:
        prdt_edited = results["prdt_edited"][pos][operation]
        if prdt_edited is not None and prdt_normal is not None:
            # 计算相对值（相对于正常token）
            ratio = prdt_edited / prdt_normal
            datas.append(ratio)
        else:
            datas.append(None)
    
    # 保存NPZ文件（如果需要）
    if output_npz:
        # 过滤掉None值，只保存有效的数据
        valid_datas = []
        valid_titles = []
        for data, title in zip(datas, titles):
            if data is not None:
                valid_datas.append(data.cpu().numpy())
                valid_titles.append(title)
        
        np.savez(
            file=os.path.join(fig_save_path, f"{operation}.npz"),
            datas=valid_datas,
            titles=valid_titles,
        )
    
    # 生成热力图（如果需要）
    if output_png:
        # 计算子图布局 (2行4列)
        n_plots = sum(1 for data in datas if data is not None)
        n_cols = 4
        n_rows = (n_plots + n_cols - 1) // n_cols  # 向上取整
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
        if n_plots == 1:
            axes = np.array([[axes]])
        elif n_rows == 1:
            axes = np.array([axes])
        
        # 展平axes数组以便迭代
        axes = axes.flatten()
        
        # 绘制有效数据
        plot_idx = 0
        for data, title in zip(datas, titles):
            if data is None:
                continue
                
            # 转换为NumPy数组（如果需要）
            if isinstance(data, torch.Tensor):
                data = data.cpu().numpy()
            
            # 绘制热力图
            im = axes[plot_idx].imshow(data, aspect='auto')
            axes[plot_idx].set_title(title)
            fig.colorbar(im, ax=axes[plot_idx], fraction=0.046, pad=0.04, label="Value")
            plot_idx += 1
        
        # 移除未使用的子图
        for i in range(plot_idx, len(axes)):
            fig.delaxes(axes[i])
        
        # 设置整体标题
        fig.suptitle(f"Attention {operation.capitalize()} Distribution", fontsize=20)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        # 保存图片
        plt.savefig(os.path.join(fig_save_path, f"{operation}.png"), dpi=dpi, bbox_inches='tight')
        plt.close(fig)
    
    print(f"Visualization for '{operation}' operation completed. "
          f"Results saved to: {os.path.abspath(fig_save_path)}")

def batch_plot_attention_results(
    results: Dict[str, Any],
    operations: List[str] = ["max", "avg"],
    fig_save_path: str = "./attention_plots",
    dpi: int = 300
) -> None:
    """
    批量生成注意力分析结果的可视化
    
    Args:
        results: AttentionAnalyzer.analyze() 返回的结果字典
        operations: 要可视化的操作类型列表
        fig_save_path: 图片保存路径
        dpi: 图片分辨率
    """
    for op in operations:
        plot_attention_results(
            results=results,
            operation=op,
            fig_save_path=fig_save_path,
            dpi=dpi
        )

def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

def test():
    config_path = "/home/yangchunhao/csc/exp/config/attn_config.json"
    config = load_json(config_path)

def main(config):
    target_generater = TargetGenerater(config.tokenizer_path, config.data_path)
    sources, predictions = target_generater.get_target()
    attention_analyzer = AttentionAnalyzer(config.attn_dir, sources, predictions)
    print(len(sources))
    print(len(predictions))
    print(sources[0])
    ananlyze_result = attention_analyzer.analyze()
    batch_plot_attention_results(
        results=ananlyze_result,
        operations=["max", "avg"],
        fig_save_path="./attention_plots",
        dpi=300
    )

@dataclass
class Config:
    attn_dir: str
    data_path: str
    tokenizer_path: str

    @classmethod
    def from_dict(cls, d):
        return cls(**d)

    @classmethod
    def from_json_file(cls, json_path):
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return cls.from_dict(data)

if __name__ == "__main__":
    """
    python /home/yangchunhao/csc/exp/analy_torch.py -config_path /home/yangchunhao/csc/exp/config/attn_config.json
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-config_path', type=str, help="config path")
    args = parser.parse_args()

    config = Config.from_json_file(args.config_path)
    main(config)