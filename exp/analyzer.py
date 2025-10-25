"""
把analyze.py封装成analyzer.py

1. 读取csc数据文件，获得文本输入(sources)输出(targets)。
2. 读取attention目录中的attention*.npz文件,获得attention分布 [layer,batch,head,seq_len,seq_len]
3. 获取观察坐标处理函数
    input: src_tgt: list[tuple(str,str)]    输入的src_tgt是经过预处理的数据，格式为[source]\t[target]
    output: list[list[tuple(int,list[int])]]  最外层每个list表示一个观察实验，第二层每个list代表每个输入，第三层的tuple表示某个source_token需要关注哪些target_token的attention score.
4. 获取观察结果处理函数
    ops={
    "max": torch.max,
    "mean": torch.mean,
    }
5. 处理观察结果
    遍历顺序为：每个观察实验、每个样例的观察坐标、每个观察结果处理函数、
    对于一个观察实验的一个样例，完整的attention结果为[layer,head,seq_len,seq_len],取其中我们需要观察的部分，张量大小为[layer, head, tgt_len]
    每个处理函数处理采样后的结果累加到数据统计数组中。
    除以样例个数得到平均值
6. 绘制热力图
    绘制含有多个子图的图表。子图网格的每行是一个观察实验，每列是一个处理函数。    
    每个热力图的含有layer*head个点，每个点代表这个head的统计结果平均值。

"""
import os
import re
import json
from tqdm import tqdm
from typing import Any, Optional
from functools import partial
from copy import deepcopy
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import torch
from transformers import AutoTokenizer, AutoConfig
from matplotlib import pyplot as plt
import seaborn as sns

import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
epsilon = 1e-5

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

class Analyzer:
    def __init__(self, attn_dir, csc_path, model_path, max_number=10000000) -> None:
        self.attn_dir = attn_dir
        self.csc_path = csc_path
        self.model_path = model_path
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model_config = AutoConfig.from_pretrained(model_path)
        self.attn_files = None
        self.src_tgt_pairs = None
        self.max_number = max_number
    
    def read_attn(self):
        """
        获取文件夹中的attention*
        因为attention文件比较大，使用时在实际读取
        """
        def natural_sort_key(s):
            # 将字符串分割为数字和非数字部分，数字转为整数用于比较
            return [int(text) if text.isdigit() else text.lower() for text in re.split(r'([0-9]+)', s)]

        files = [f for f in os.listdir(self.attn_dir) if f.startswith(f'attention')]
        self.attn_files = sorted(files, key=natural_sort_key)[:self.max_number]
        return self.attn_files

    def read_csc(self):
        with open(self.csc_path, 'r', encoding='utf-8') as f:
            self.csc_data = json.load(f)
        
        self.src_tgt_pairs = [(x['source'], x['target']) for x in self.csc_data['metadata']['result']]
    
    def plot_attentions(self, exp_metadata: list[dict[str, Any]], titles, save_path):
        """
        每个实验一行（如观察第一个错别字到原字符的attentoin score)，每列代表一种处理函数(如：max, mean)
        """
        fig, axes = plt.subplots(len(exp_metadata[0]), len(exp_metadata), figsize=(len(exp_metadata[0]) * 5, len(exp_metadata) * 5))
        axes = axes.flatten()
        
        datas = [metadata[op_name].cpu() for metadata in exp_metadata for op_name in metadata.keys()]

        if not os.path.exists(save_path):
            os.makedirs(save_path)
        np.savez(
            file=os.path.join(save_path, "analysis_res.npz"),
            datas = datas,
            titles = titles,
        )
        for ax, data, title in zip(axes, datas, titles):
            im = ax.imshow(data)
            ax.set_title(title)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="value")

        plt.tight_layout()
        plt.savefig(os.path.join(save_path, "attention_feature.png"), dpi=300)
        plt.close(fig)

    def str2token_list(self, s: str)->list[list[str]]:
        """
        对输入str进行分词
        """
        return [self.tokenizer.decode(t) for t in self.tokenizer.encode(s)]
    
    def analyze(self, exp_position_functions: dict[str, Any], ops: dict[str, Any], save_path: str):
        self.read_attn()
        self.read_csc()
        src_tgt_tokens = [
            (self.str2token_list(src), self.str2token_list(tgt))
            for src, tgt in self.src_tgt_pairs
        ]
        exp_positions = [
            [
                exp_position_function(src_tokens, tgt_tokens)
                for src_tokens, tgt_tokens in src_tgt_tokens
            ] for exp_position_function in exp_position_functions.values()
        ]
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        num_hidden_layers, num_attention_heads = self.model_config.num_hidden_layers, self.model_config.num_attention_heads
        exp_metadata = [
            {
                op: torch.zeros((num_hidden_layers, num_attention_heads), dtype=torch.float64, device=device)
                for op in ops.keys()
            }
            for _ in range(len(exp_position_functions))
        ]
        exp_position_counts = torch.zeros(len(exp_position_functions))
        for filename in tqdm(self.attn_files, desc="Attention analysis"):
            attn_metadata = np.load(os.path.join(self.attn_dir, filename))
            start_id = attn_metadata['start_id']
            end_id = attn_metadata['end_id']
            
            for idx in range(start_id, end_id):
                attn_matrix = torch.from_numpy(attn_metadata['attentions'][:, idx - start_id]).to(device)
                has_abnormal = (~torch.isfinite(attn_matrix)).any()
                if has_abnormal:
                    print(f"Found abnormal attention matrix at {idx}")
                temp_exp_metadata = [
                    {
                        op: torch.zeros((num_hidden_layers, num_attention_heads), dtype=torch.float64, device=device)
                        for op in ops.keys()
                    }
                    for _ in range(len(exp_position_functions))
                ]
                for eid, e_pos_list in enumerate(exp_positions):
                    temp_exp_count = 0
                    for qid, kids in e_pos_list[idx]:
                        if qid == None:
                            continue
                        temp_exp_count += 1
                        for op_name, op in ops.items():
                            # print(op(attn_matrix[:, :, qid, kids]).shape)
                            temp_exp_metadata[eid][op_name] += op(attn_matrix[:, :, qid, kids])
                            
                    
                    # 对一个句子中的一类位置进行归一化
                    if temp_exp_count > 0:
                        exp_position_counts[eid] += 1
                        for op_name, op in ops.items():
                            temp_exp_metadata[eid][op_name] /= temp_exp_count

                # # 所有实验 和 normal token 作对比
                # for metadata in temp_exp_metadata[1:]:
                #     for op in metadata.keys():
                #         metadata[op] /= (temp_exp_metadata[0][op] + epsilon)
                #         if torch.max(metadata[op]) > 1000:
                #             idx += 1
                #             idx -= 1

                # 累加到总实验数据中
                for metadata, metadata_ in zip(exp_metadata, temp_exp_metadata):
                    # 某些干扰值去除掉
                    if torch.max(metadata_[op_name]) >= 50:
                        with open("./exp_log.md", 'a') as f:
                            print(f"WARNING: {op_name}, at idx = {idx}" , file=f)
                        continue
                    for op_name in metadata.keys():
                        metadata[op_name] += metadata_[op_name]

        # 对每个token对应 attention 进行 op 运算后的结果取平均值
        for metadata, exp_p in zip(exp_metadata, exp_position_counts):
            for op_name in metadata.keys():
                metadata[op_name] /= exp_p
        
            # # TODO:某些情况下可以使用mask掩码来批处理数据，可能能大幅加快处理速度？
            # attn_matrix = torch.from_numpy(attn_metadata['attentions'], device=device)
            # # 生成mask掩码
            # for positions_list in exp_positions:
            #     mask = torch.zeros_like(attn_matrix[0, :, :, :], dtype=torch.bool, device=device)
            #     for id in range(start_id, end_id):
            #         positions = positions_list[id]
            #         if positions == None
            #         for qid, kids in positions:
            #             # <im_end>\n<im_start>assistant\n是6个token，对于qwen系列都是这样，直接跳过
            #             kids_tensor = torch.tensor(kids, dtype=torchint, device=device) + qid + 6
            #             mask[id, qid, kids_tensor] = True
                
            #     selected_attn = attn_matrix[:, mask]

        titles = [exp_name + '_' + op_name for exp_name in exp_position_functions.keys() for op_name in ops.keys()]
        self.plot_attentions(exp_metadata, titles, save_path)
        return exp_metadata
    
    def plot_precision_recall_vs_threshold(self, normal_dis, exp_dis, title="Precision-Recall vs. Threshold", save_path=None):
        """
        根据给定的正常和异常分布的分数，绘制Precision和Recall随阈值变化的曲线。
        并标记使 F1 分数最大的阈值点（最佳阈值）。

        Args:
            normal_dis (array-like): 正常类别（负例，label=0）的分数/距离分布。
            exp_dis (array-like): 异常类别（正例，label=1）的分数/距离分布。
            title (str): 图表的标题。
            save_path (str, optional): 保存图表的路径。如果为None，则不保存。
        """
        import numpy as np
        import matplotlib.pyplot as plt
        from sklearn.metrics import precision_recall_curve
        import os

        # 1. 准备真实标签和对应的分数
        y_true = np.concatenate([np.zeros(len(normal_dis)), np.ones(len(exp_dis))])
        y_scores = np.concatenate([normal_dis, exp_dis])

        # 2. 使用 scikit-learn 计算 precision, recall 和 thresholds
        precision, recall, thresholds = precision_recall_curve(y_true, y_scores)

        # --- 3. 开始绘图 ---
        plt.figure(figsize=(9, 6))

        # 注意：precision 和 recall 比 thresholds 多一个元素（最后一个对应 threshold=-inf）
        # 所以我们使用 thresholds 作为 x 轴，并去掉 precision 和 recall 的最后一个元素
        prec = precision[:-1]
        rec = recall[:-1]
        thres = thresholds

        # ✅ 新增：计算 F1 分数
        f1_scores = 2 * (prec * rec) / (prec + rec + 1e-8)  # 加上极小值避免除零

        # ✅ 找到 F1 最大值对应的索引
        best_f1_idx = np.argmax(f1_scores)
        best_threshold_f1 = thres[best_f1_idx]
        best_precision = prec[best_f1_idx]
        best_recall = rec[best_f1_idx]
        best_f1 = f1_scores[best_f1_idx]

        # 绘制 Precision 和 Recall 曲线
        plt.plot(thres, prec, 'b--', label='Precision')
        plt.plot(thres, rec, 'g-', label='Recall')

        # ✅ 在图上标记 F1 最佳点
        plt.plot(best_threshold_f1, best_precision, 'ro', ms=8, 
                label=f'Best F1 (θ≈{best_threshold_f1:.2f}, F1={best_f1:.3f}), precision={best_precision:.3f}, recall={best_recall:.3f}')
        plt.axvline(x=best_threshold_f1, color='red', linestyle=':', linewidth=1)

        # 可选：同时标出 Precision=Recall 交点（如果你还想保留它）
        intersection_idx = np.argmin(np.abs(prec - rec))
        best_threshold_intersect = thres[intersection_idx]
        plt.plot(best_threshold_intersect, prec[intersection_idx], 'mo', ms=6, 
                label=f'P=R (θ≈{best_threshold_intersect:.2f})')
        plt.axvline(x=best_threshold_intersect, color='magenta', linestyle='--', linewidth=1)

        # --- 美化图表 ---
        plt.title(title, fontsize=16)
        plt.xlabel('Threshold (θ)', fontsize=12)
        plt.ylabel('Score', fontsize=12)
        plt.legend(loc='best')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.xlim([0.0, 1.0])  # 假设分数范围在 [0,1]，否则可注释或动态调整
        plt.ylim([0.0, 1.05])

        # 4. 保存和显示
        if save_path:
            save_dir = os.path.dirname(save_path)
            if save_dir and not os.path.exists(save_dir):
                os.makedirs(save_dir)
            plt.savefig(save_path)
            print(f"Plot saved to {save_path}")

        plt.show()
        plt.close()
        
    def plot_distance(self, statistic, titles, save_path):
        nrows=len(statistic) - 1
        ncols=len(statistic[0])
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(7 * ncols, 5 * nrows))
        # 将axes展平成一维数组，方便遍历
        if nrows * ncols > 1:
            axes = axes.flatten()
        else:
            axes = np.array([axes])
        # 遍历每个实验（第一个不遍历，是normal token）
        for eid, exp_data in enumerate(statistic[1:]):
            # 遍历距离函数
            for d_id, data in enumerate(exp_data):
                normal_dis = statistic[0][d_id]
                exp_dis = exp_data[d_id]
            
                ax = axes[eid * ncols + d_id]
            
                # # 使用 seaborn 绘制KDE图
                # sns.histplot(data=normal_dis, ax=ax, label='normal', color='royalblue', alpha=0.5, 
                #          kde=True, stat="density", bins=50, 
                #          kde_kws={'clip': (0.0, 1.0)}) # Clip KDE at 0

                # # Plot for the 'error' distribution
                # sns.histplot(data=exp_dis, ax=ax, label='error', color='darkorange', alpha=0.5,
                #             kde=True, stat="density", bins=50,
                #             kde_kws={'clip': (0.0, 1.0)}) # Clip KDE at 0
                # # --- 美化图表 ---
                # ax.set_title(titles[eid * ncols + d_id], fontsize=14, fontweight='bold')
                # ax.set_xlabel('Calculated Distance', fontsize=12)
                # ax.set_ylabel('Density', fontsize=12)
                # ax.legend()
                # ax.grid(axis='y', linestyle='--', alpha=0.7)

                sns.histplot(
                    data=normal_dis,
                    ax=ax,
                    color='royalblue',
                    alpha=0.7,
                    label='normal',
                    stat='density',  # 确保与KDE刻度一致
                    # bins=200,         # 适当数量的bins
                    element='step',  # 线框样式
                    fill=False,
                    linewidth=1.5
                )
                sns.histplot(
                    data=exp_dis,
                    ax=ax,
                    color='darkorange',
                    alpha=0.7,
                    label='error',
                    stat='density',
                    # bins=200,
                    element='step',
                    fill=False,
                    linewidth=1.5
                )
                
                # # 2. 绘制KDE（带边界修正）
                # sns.kdeplot(
                #     data=normal_dis,
                #     ax=ax,
                #     color='royalblue',
                #     fill=True,
                #     alpha=0.3,  # 降低KDE填充透明度
                #     clip=(0, None),  # 关键：修正负值问题
                #     label='_nolegend_',  # 避免重复图例
                #     bw_method=0.1  # 减小带宽，使KDE更贴合数据
                # )
                # sns.kdeplot(
                #     data=exp_dis,
                #     ax=ax,
                #     color='darkorange',
                #     fill=True,
                #     alpha=0.3,
                #     clip=(0, None),
                #     label='_nolegend_',
                #     bw_method=0.1
                # )
                
                # ===== 其他优化 =====
                ax.set_xlim(0, None)  # 强制x轴从0开始
                ax.set_title(titles[eid * ncols + d_id], fontsize=14, fontweight='bold')
                ax.set_xlabel('Calculated Distance', fontsize=12)
                ax.set_ylabel('Density', fontsize=12)
                
                # 自定义图例（避免直方图和KDE重复）
                from matplotlib.lines import Line2D
                legend_elements = [
                    Line2D([0], [0], color='royalblue', lw=2, label='normal (hist)'),
                    Line2D([0], [0], color='darkorange', lw=2, label='error (hist)'),
                    Line2D([0], [0], color='royalblue', lw=4, alpha=0.3, label='normal (KDE)'),
                    Line2D([0], [0], color='darkorange', lw=4, alpha=0.3, label='error (KDE)')
                ]
                ax.legend(handles=legend_elements)
                
                ax.grid(axis='y', linestyle='--', alpha=0.7)
        
        plt.tight_layout(pad=3.0)
        plt.show()
        plt.savefig(save_path, dpi=600)
        plt.close()

    def static_distance_from_attn(self, exp_attention, distance_functions: dict[str, Any], standard, op, save_path: str):
        exp_static_distance = [
            [
                [] for _ in range(len(distance_functions))
            ]
            for i in range(len(exp_attention.keys()))
        ]
        for e_id, (exp_name, attn_path) in enumerate(exp_attention.items()):
            attn = np.load(attn_path)
            for token_attn in attn:
                for i, distance_function in enumerate(distance_functions.values()):
                    exp_static_distance[e_id][i].append(distance_function(op(token_attn), standard))
        
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        titles = [exp_name + '_' + distance_function_name for exp_name in exp_attention.keys() for distance_function_name in distance_functions.keys()]
        self.plot_distance(exp_static_distance, titles, save_path)
        self.plot_precision_recall_vs_threshold(exp_static_distance[0][0], exp_static_distance[1][0], save_path=save_path / "precision_recall.png")

    def static_distance(self, exp_position_functions: dict[str, Any], exp_attention, distance_functions: dict[str, Any], standard, op, save_path: str):
        if exp_attention is not None:
            self.static_distance_from_attn(exp_attention, distance_functions, standard, op, save_path)
            return
        
        # TODO:
        self.read_attn()
        self.read_csc()
        src_tgt_tokens = [
            (self.str2token_list(src), self.str2token_list(tgt))
            for src, tgt in self.src_tgt_pairs
        ]
        exp_positions = [
            [
                exp_position_function(src_tokens, tgt_tokens)
                for src_tokens, tgt_tokens in src_tgt_tokens
            ] for exp_position_function in exp_position_functions.values()
        ]
        statistic = [
            [
                [] for func in len(distance_functions)
            ]
            for exp_position in exp_positions
        ]
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        num_hidden_layers, num_attention_heads = self.model_config.num_hidden_layers, self.model_config.num_attention_heads

        for filename in tqdm(self.attn_files, desc="Statistic distance"):
            attn_metadata = np.load(os.path.join(self.attn_dir, filename))
            start_id = attn_metadata['start_id']
            end_id = attn_metadata['end_id']
            
            for idx in range(start_id, end_id):
                attn_matrix = attn_metadata['attentions'][:, idx - start_id]
                for eid, e_pos_list in enumerate(exp_positions):
                    temp_exp_count = 0
                    for qid, kids in e_pos_list[idx]:
                        if qid == None:
                            continue
                        
                        for distance_func_id, distance_func in enumerate(distance_functions.values()): 
                            cur_attn = op(attn_matrix[:, :, qid, kids])
                            statistic[eid][distance_func_id].append(distance_func(cur_attn.flatten(), standard.flatten()))

        titles = [exp_name + '_' + distance_function_name for exp_name in exp_position_functions.keys() for distance_function_name in distance_functions.keys()]
        self.plot_distance(statistic, titles, save_path)
    
    def pre_extract(self, exp_position_functions: dict[str, Any], save_path: str, max_workers: Optional[int] = None):
        self.read_attn()
        self.read_csc()
        src_tgt_tokens = [
            (self.str2token_list(src), self.str2token_list(tgt))
            for src, tgt in self.src_tgt_pairs
        ]
        exp_positions = [
            [
                exp_position_function(src_tokens, tgt_tokens)
                for src_tokens, tgt_tokens in src_tgt_tokens
            ] for exp_position_function in exp_position_functions.values()
        ]
        
        exp_names = list(exp_position_functions.keys())
        exp_attn_matrix: dict[str, list[tuple[int, int, np.ndarray]]] = {
            exp_name: [] for exp_name in exp_names
        }

        if max_workers is None:
            cpu_cnt = os.cpu_count() or 1
            max_workers = min(cpu_cnt, len(self.attn_files)) or 1
        else:
            max_workers = max(1, min(max_workers, len(self.attn_files)))

        def process_file(filename: str) -> dict[str, list[tuple[int, int, np.ndarray]]]:
            file_path = os.path.join(self.attn_dir, filename)
            local_results: dict[str, list[tuple[int, int, np.ndarray]]] = {
                exp_name: [] for exp_name in exp_names
            }
            with np.load(file_path) as attn_metadata:
                start_id = int(attn_metadata['start_id'])
                end_id = int(attn_metadata['end_id'])
                attentions = attn_metadata['attentions']
                for idx in range(start_id, end_id):
                    attn_matrix = attentions[:, idx - start_id]
                    for eid, e_pos_list in enumerate(exp_positions):
                        for pos_idx, (qid, kids) in enumerate(e_pos_list[idx]):
                            if qid is None:
                                continue
                            local_results[exp_names[eid]].append(
                                (idx, pos_idx, attn_matrix[:, :, qid, kids])
                            )
            return local_results

        if max_workers == 1:
            iterable = self.attn_files
            pbar = tqdm(iterable, desc="Pre Extract attention")
            for filename in pbar:
                file_results = process_file(filename)
                for exp_name, values in file_results.items():
                    exp_attn_matrix[exp_name].extend(values)
        else:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(process_file, filename): filename
                    for filename in self.attn_files
                }
                for future in tqdm(as_completed(futures), total=len(futures), desc="Pre Extract attention"):
                    file_results = future.result()
                    for exp_name, values in file_results.items():
                        exp_attn_matrix[exp_name].extend(values)
        
        # save_path不存在则创建目录
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        for exp_name in exp_position_functions.keys():
            ordered = sorted(exp_attn_matrix[exp_name], key=lambda item: (item[0], item[1]))
            tensors = [item[2] for item in ordered]
            if tensors:
                np.save(os.path.join(save_path, f"{exp_name}.npy"), np.stack(tensors))
            else:
                np.save(os.path.join(save_path, f"{exp_name}.npy"), np.empty((0,)))

def find_first_diff(src_tokens, tgt_tokens) -> list[tuple[Any, list[Any]]]:
    """
    找到第一个不同的token位置p，我们要获取生成p时向原字符的attention score。
    """
    # 添加一个pad字符用来用来处理最后一个字缺失的问题
    src_tokens_pad = deepcopy(src_tokens) + ["<im_end>"]
    tgt_tokens_pad = deepcopy(tgt_tokens) + ["<im_end>"]
    src_len = len(src_tokens)
    for i in range(min(len(src_tokens_pad), len(tgt_tokens_pad))):
        if src_tokens_pad[i] != tgt_tokens_pad[i]:
            return [(i-1 + src_len+5, [i])]

    return [(None, [None])]


def get_normal_token(src_tokens, tgt_tokens) -> list[tuple[str, list[str]]]:
    """
    获取第一个不同token之前的token
    """
    # 添加一个pad字符用来用来处理最后一个字缺失的问题
    exp_positions = []
    src_tokens_pad = deepcopy(src_tokens) + ["<im_end>"]
    tgt_tokens_pad = deepcopy(tgt_tokens) + ["<im_end>"]
    src_len = len(src_tokens)
    for i in range(min(len(src_tokens_pad), len(tgt_tokens_pad))):
        if src_tokens_pad[i] == tgt_tokens_pad[i]:
            exp_positions.append((i + src_len + 5, list(range(len(src_tokens)))))
        else:
            break
    
    if exp_positions == []:
        return [(None, [None])]
    return exp_positions

def get_normal_token_coresponding(src_tokens, tgt_tokens) -> list[tuple[str, list[str]]]:
    """
    获取生成target时和sorce相同Token的attention score
    """
    # 添加一个pad字符用来用来处理最后一个字缺失的问题
    exp_positions = []
    src_tokens_pad = deepcopy(src_tokens) + ["<im_end>"]
    tgt_tokens_pad = deepcopy(tgt_tokens) + ["<im_end>"]
    src_len = len(src_tokens)
    # sorces_tokens <im_end> \n <im_start> assistant \n target_tokens
    for i in range(min(len(src_tokens_pad), len(tgt_tokens_pad))):
        if src_tokens_pad[i] == tgt_tokens_pad[i]:
            exp_positions.append((i - 1 + src_len + 5, [i]))
        else:
            break
    
    if exp_positions == []:
        return [(None, [None])]
    return exp_positions

def first_diff_p2minus_k(src_tokens, tgt_tokens, k=0) -> list[tuple[Any, list[Any]]]:
    """
    深层时token向p-2的位置
    找到第一个不同的token位置p，我们要获取生成p时向原字符的attention score。
    """
    # 添加一个pad字符用来用来处理最后一个字缺失的问题
    src_tokens_pad = deepcopy(src_tokens) + ["<im_end>"]
    tgt_tokens_pad = deepcopy(tgt_tokens) + ["<im_end>"]
    src_len = len(src_tokens)
    for i in range(min(len(src_tokens_pad), len(tgt_tokens_pad))):
        if src_tokens_pad[i] != tgt_tokens_pad[i]:
            # p->p-k 所以p-1 -> p-k-1
            return [(i-1 + src_len+5, [i-k-1])]

    return [(None, [None])]

def normal_token_p2minus_k(src_tokens, tgt_tokens, k=0) -> list[tuple[str, list[str]]]:
    """
    获取第一个不同token之前的token，向p-2位置
    """
    # 添加一个pad字符用来用来处理最后一个字缺失的问题
    exp_positions = []
    src_tokens_pad = deepcopy(src_tokens) + ["<im_end>"]
    tgt_tokens_pad = deepcopy(tgt_tokens) + ["<im_end>"]
    src_len = len(src_tokens)
    for i in range(min(len(src_tokens_pad), len(tgt_tokens_pad))):
        if src_tokens_pad[i] == tgt_tokens_pad[i]:
            # p->p-k 所以p-1 -> p-k-1
            exp_positions.append((i + src_len + 5, [i-k-1]))
        else:
            break
    
    if exp_positions == []:
        return [(None, [None])]
    return exp_positions


def config_analyze():
    init_config = {
        "attn_dir": "/home/yangchunhao/csc/exp/attn_target",
        "csc_path": "/home/yangchunhao/csc/results/Qwen2.5-VL-7B-Instruct_cscd-ns-test/checkpoints/csc.json",
        "model_path": "/data/images/llms/Qwen/Qwen2.5-VL-7B-Instruct",
        # "max_number": 20
    }
#   默认第0行是normal token。所有其他token的attention值都需要和normal token的值作对比
    exp_position_functions = {
        "normal_token": get_normal_token_coresponding,
        "first_diff_token": find_first_diff,
    }
    ops = {
        "max": partial(torch.amax, dim=-1),
        "mean": partial(torch.mean, dim=-1),
    }
    analyze_config = {
        "save_path": "/home/yangchunhao/csc/exp/analy_res/test",
        "ops": ops,
        "exp_position_functions": exp_position_functions,
    }

    return init_config, analyze_config

def config_distance():
    init_config = {
        "attn_dir": "/home/yangchunhao/csc/exp/attn_target",
        "csc_path": "/home/yangchunhao/csc/results/Qwen2.5-VL-7B-Instruct_cscd-ns-test/checkpoints/csc.json",
        "model_path": "/data/images/llms/Qwen/Qwen2.5-VL-7B-Instruct",
        "max_number": 20
    }
    exp_position_functions = {
        "normal_token": get_normal_token_coresponding,
        "first_diff_token": find_first_diff,
    }
    exp_attention = {
        "normal_token": "/home/yangchunhao/csc/exp/p2p/normal_token.npy",
        "first_diff_token": "/home/yangchunhao/csc/exp/p2p/first_diff_token.npy",
    }
    from scipy.spatial.distance import cosine, jensenshannon
    distance_functions = {
        # "jensenshannon" : jensenshannon,
        # "cosine": cosine,
        "layer18head14": lambda x,y: x[18,14] - y,
    }
    # 注意时用torch还是numpy
    # op = partial(torch.amax, dim=-1)
    op = partial(np.amax, axis=-1)

    standard_path = "/home/yangchunhao/csc/exp/analy_res/test/analysis_res.npz"
    standard = np.load(standard_path)
    distance_config = {
        "exp_position_functions": exp_position_functions,
        "exp_attention" : exp_attention,
        "distance_functions": distance_functions,
        "op": op,
        "standard": 0,
        "save_path": "/home/yangchunhao/csc/exp/analy_res/test2",
    }

    return init_config, distance_config

def config_pre_extract():
    # 抽取某些位置的attention，避免每次处理数据时加载大批量数据
    init_config = {
        "attn_dir": "/home/yangchunhao/csc/exp/attn/cscd-ns_train_vl_step/src_tgt",
        "csc_path": "/home/yangchunhao/csc/results/Qwen2.5-VL-7B-Instruct_cscd-ns-train_step/checkpoints/csc.json",
        "model_path": "/data/images/llms/Qwen/Qwen2.5-VL-7B-Instruct",
        "max_number": 5000,
    }
    exp_position_functions = {
        "normal_token": get_normal_token_coresponding,
        "first_diff_token": find_first_diff,
    }
    extract_config = {
        "exp_position_functions" : exp_position_functions,
        "save_path": "/home/yangchunhao/csc/exp/p2p/cscd-ns/train",
        "max_workers": 8,
    }

    # init_config = {
    #     "attn_dir": "/home/yangchunhao/csc/exp/attn/cscd-ns_test_vl_step/src_tgt",
    #     "csc_path": "/home/yangchunhao/csc/results/Qwen2.5-VL-7B-Instruct_cscd-ns-test_step/checkpoints/csc.json",
    #     "model_path": "/data/images/llms/Qwen/Qwen2.5-VL-7B-Instruct",
    #     "max_number": 5000,
    # }
    # exp_position_functions = {
    #     "normal_token": get_normal_token_coresponding,
    #     "first_diff_token": find_first_diff,
    # }
    # extract_config = {
    #     "exp_position_functions" : exp_position_functions,
    #     "save_path": "/home/yangchunhao/csc/exp/p2p/cscd-ns/test",
    #     "max_workers": 8,
    # }
    return init_config, extract_config

def analyze():
    init_config, analyze_config = config_analyze()
    analyzer = Analyzer(**init_config)
    analyzer.analyze(**analyze_config)

def statistic_distance():
    init_config, distance_config = config_distance()
    analyzer = Analyzer(**init_config)
    analyzer.static_distance(**distance_config)

def pre_extract():
    init_config, extract_config = config_pre_extract()
    analyzer = Analyzer(**init_config)
    analyzer.pre_extract(**extract_config)

if __name__ == "__main__":
    pre_extract()
    # analyze()
    # statistic_distance()
