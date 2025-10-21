"""
中文拼写纠错评估模块
提供字符级和句子级的precision, recall, F1指标计算
"""

import json
import os
import logging
from typing import Dict, List, Tuple, Any
from collections import Counter
import numpy as np
from pathlib import Path
from string import punctuation
from tqdm import tqdm

from utils.file_utils import read_csc_data, load_results
from utils.logging_utils import get_logger
try:
    from .utils import input_check_and_process, compute_p_r_f1, write_report, Alignment
except ImportError:
    from utils import input_check_and_process, compute_p_r_f1, write_report, Alignment

logger = get_logger(__name__)


class Evaluator:
    """评估器类，用于计算CSC任务的性能指标"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化评估器
        
        Args:
            config: 评估配置字典
        """
        self.config = config
        self.metrics = {}
        
    def load_data(self, source_path: str, target_path: str) -> Tuple[List[str], List[str]]:
        """
        加载源数据和目标数据
        
        Args:
            source_path: 源数据文件路径
            target_path: 目标数据文件路径
            
        Returns:
            (sources, targets) 元组
        """
        try:
            if source_path and target_path:
                sources, targets = read_csc_data(source_path, target_path)
            else:
                # 从结果文件中加载
                results = load_results(self.config.get('result_file', ''))
                sources = [item['source'] for item in results]
                targets = [item['target'] for item in results]
                
            logger.info(f"加载了 {len(sources)} 条数据用于评估")
            return sources, targets
            
        except Exception as e:
            logger.error(f"加载数据失败: {e}")
            raise
    
    def load_predictions(self, result_file: str) -> List[str]:
        """
        加载模型预测结果
        
        Args:
            result_file: 结果文件路径
            
        Returns:
            预测结果列表
        """
        try:
            results = load_results(result_file)
            predictions = [item['prediction'] for item in results]
            logger.info(f"加载了 {len(predictions)} 条预测结果")
            return predictions
            
        except Exception as e:
            logger.error(f"加载预测结果失败: {e}")
            raise
    
    # Adapted from: https://github.com/nghuyong/cscd-ime/blob/master/evaluation/evaluate.py
    def calculate_metric(self, src_sentences, tgt_sentences, pred_sentences, report_file=None, ignore_chars=""):
        """
        :param src_sentences: list of origin sentences
        :param tgt_sentences: list of target sentences
        :param pred_sentences: list of predict sentences
        :param report_file: report file path
        :param ignore_chars: chars that is not evaluated
        :return:
        """
        src_char_list, tgt_char_list, pred_char_list = input_check_and_process(src_sentences, tgt_sentences, pred_sentences)
        sentence_detection, sentence_correction, char_detection, char_correction = [
            {'all_error': 0, 'true_predict': 0, 'all_predict': 0} for _ in range(4)]
        n_not_error = 0
        n_false_pred = 0
        output_errors = []
        for src_chars, tgt_chars, pred_chars in tqdm(zip(src_char_list, tgt_char_list, pred_char_list), total=len(pred_char_list)):
            true_error_indexes = set()
            true_error_edits = set()
            detect_indexes = set()
            detect_edits = set()

            gold_edits = Alignment(src_chars, tgt_chars).align_seq
            pred_edits = Alignment(src_chars, pred_chars).align_seq

            for gold_edit in gold_edits:
                edit_type, b_ori, e_ori, b_prd, e_prd = gold_edit
                if edit_type != 'M':
                    src_char = ''.join(src_chars[b_ori:e_ori])
                    if len(src_char) > 0 and src_char in ignore_chars:
                        continue
                    char_detection['all_error'] += 1
                    char_correction['all_error'] += 1
                    true_error_indexes.add((b_ori, e_ori))
                    true_error_edits.add((b_ori, e_ori, tuple(tgt_chars[b_prd:e_prd])))

            for pred_edit in pred_edits:
                edit_type, b_ori, e_ori, b_prd, e_prd = pred_edit
                if edit_type != 'M':
                    src_char = ''.join(src_chars[b_ori:e_ori])
                    if len(src_char) > 0 and src_char in ignore_chars:
                        continue
                    char_detection['all_predict'] += 1
                    char_correction['all_predict'] += 1
                    detect_indexes.add((b_ori, e_ori))
                    detect_edits.add((b_ori, e_ori, tuple(pred_chars[b_prd:e_prd])))
                    if (b_ori, e_ori) in true_error_indexes:
                        char_detection['true_predict'] += 1
                    if (b_ori, e_ori, tuple(pred_chars[b_prd:e_prd])) in true_error_edits:
                        char_correction['true_predict'] += 1

            if true_error_indexes:
                sentence_detection['all_error'] += 1
                sentence_correction['all_error'] += 1
            else:
                n_not_error += 1
                if detect_indexes:
                    n_false_pred += 1
            if detect_indexes:
                sentence_detection['all_predict'] += 1
                sentence_correction['all_predict'] += 1
                if tuple(true_error_indexes) == tuple(detect_indexes):
                    sentence_detection['true_predict'] += 1
                if tuple(true_error_edits) == tuple(detect_edits):
                    sentence_correction['true_predict'] += 1

            # origin_s = "".join(src_chars)
            # target_s = "".join(tgt_chars)
            # predict_s = "".join(pred_chars)
            # if target_s == predict_s:
            #     error_type = "正确"
            # elif origin_s == target_s and origin_s != predict_s:
            #     error_type = "过纠"
            # elif origin_s != target_s and origin_s == predict_s:
            #     error_type = "漏纠"
            # else:
            #     error_type = '综合'
            # print(f"detect_edits: {detect_edits}")
            # print(f"true_error_edits: {true_error_edits}")
            if true_error_edits == detect_edits:
                error_type = "正确"
            elif (true_error_edits == set() and detect_edits != set()) or all([edit in detect_edits for edit in true_error_edits]):
                error_type = "过纠"
            elif (true_error_edits != set() and detect_edits == set()) or all([edit in true_error_edits for edit in detect_edits]):
                error_type = "漏纠"
            else:
                error_type = '综合'
            output_errors.append(
                [
                    "原始: " + "".join(src_chars),
                    "正确: " + "".join(["".join(tgt_chars[t_b:t_e]) if (s_b, s_e) not in true_error_indexes else f"【{''.join(tgt_chars[t_b:t_e])}】" for _, s_b, s_e, t_b, t_e in gold_edits]),
                    "预测: " + "".join(["".join(pred_chars[t_b:t_e]) if (s_b, s_e) not in detect_indexes else f"【{''.join(pred_chars[t_b:t_e])}】" for _, s_b, s_e, t_b, t_e in pred_edits]),
                    "错误类型: " + error_type,
                ]
            )

        result = dict()
        prefix_names = [
            "sentence detection ",
            "sentence correction ",
            "char detection ",
            "char correction ",
        ]
        for prefix_name, sub_metric in zip(prefix_names,
                                        [sentence_detection, sentence_correction, char_detection, char_correction]):
            sub_r = compute_p_r_f1(sub_metric['true_predict'], sub_metric['all_predict'], sub_metric['all_error']).items()
            for k, v in sub_r:
                result[prefix_name + k] = v
        
        result["sentence fpr"] = round(100 * n_false_pred / (n_not_error + 1e-10), 3)

        if report_file:
            write_report(report_file, result, output_errors)
        return result
    
    def evaluate(self, 
                sources: List[str] = None, 
                targets: List[str] = None, 
                predictions: List[str] = None) -> Dict[str, Any]:
        """
        执行完整评估流程
        
        Args:
            sources: 原始文本列表（可选）
            targets: 正确文本列表（可选）
            predictions: 预测文本列表（可选）
            
        Returns:
            评估结果字典
        """
        try:
            # 如果没有提供数据，从配置文件加载
            if sources is None or targets is None:
                source_path = self.config.get('source_file')
                target_path = self.config.get('target_file')
                sources, targets = self.load_data(source_path, target_path)
            
            if predictions is None:
                result_file = self.config.get('result_file')
                predictions = self.load_predictions(result_file)
            
            # 验证数据集长度
            if not (len(sources) == len(targets) == len(predictions)):
                raise ValueError(
                    f"数据长度不匹配: sources={len(sources)}, "
                    f"targets={len(targets)}, predictions={len(predictions)}"
                )
            
            chars_to_ignore = set(self.config.get('ignore_chars', ""))
            # 配置忽略计算的字符
            if self.config.get('ignore_punct', True):
                chinese_punct = "！？｡＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘'‛“”„‟…‧﹏"
                english_punct = punctuation
                punct_set = set(chinese_punct + english_punct)
                chars_to_ignore = chars_to_ignore.union(punct_set)
            
            if self.config.get('output_dir', None):
                report_file = os.path.join(self.config.get('output_dir', None), "report.txt")
            else:
                report_file = None

            # 计算评估指标
            metrics = self.calculate_metric(sources, targets, predictions, report_file, chars_to_ignore)

            print(f"评估指标: {json.dumps(metrics, ensure_ascii=False, indent=2)}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"评估失败: {e}")
            raise


def main():
    """评估器的主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='中文拼写纠错评估工具')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                        help='配置文件路径')
    parser.add_argument('--source', type=str, help='源数据文件路径')
    parser.add_argument('--target', type=str, help='目标数据文件路径')
    parser.add_argument('--result', type=str, help='预测结果文件路径')
    parser.add_argument('--output', type=str, help='评估结果输出文件路径')
    
    args = parser.parse_args()
    
    # 加载配置
    from utils.file_utils import load_config
    config = load_config(args.config)
    
    # 更新配置
    if args.source:
        config['evaluation']['source_file'] = args.source
    if args.target:
        config['evaluation']['target_file'] = args.target
    if args.result:
        config['evaluation']['result_file'] = args.result
    if args.output:
        config['evaluation']['output_file'] = args.output
    
    # 执行评估
    evaluator = Evaluator(config['evaluation'])
    metrics = evaluator.evaluate()
    
    return metrics

def test():
    # sources = ["桃花庄曾是一片荒山,阒无人烟。"]
    # targets = ["桃花庄曾是一片荒山,阒无人烟。"]
    # predictions = ["桃花庄曾是一片荒山，阒无人烟。"]
    # sources = ["国棉厂长途快运怎么收费其实无论是办公家具公司送还是个人自己搬回去都要注意一下几点:首先是搬运车辆的选择。"]
    # targets = ["国棉厂长途快运怎么收费其实无论是办公家具公司送还是个人自己搬回去都要注意以下几点:首先是搬运车辆的选择。"]
    # predictions = ["国棉厂长途快运怎么收费其实无论是办公家具公司送还是个人自己搬回去都要注意一下几点:首先是搬运车辆的选择。"]
    sources = ["我买的是情怀，但是老罗你确深深的欺骗了你的用户，这种欺骗无法饶恕。"]
    targets = ["我买的是情怀，但是老罗你却深深地欺骗了你的用户，这种欺骗无法饶恕。"]
    predictions = ["我买的是情怀，但是老罗你却深深的欺骗了你的用户，这种欺骗无法饶恕。"]
    sources = ["将来有了孩子，从幼儿园开始，就是无止境的择校费赞助费，临到高考还可能被遣回原籍……对异乡人而言，北京就想个无边的巨大漩涡，你只有被裹挟着一圈圈打转，不知何时才能停下来喘口气"]
    targets = ["将来有了孩子，从幼儿园开始，就是无止境的择校费赞助费，临到高考还可能被遣回原籍……对异乡人而言，北京就像个无边的巨大漩涡，你只有被裹挟着一圈圈打转，不知何时才能停下来喘口气"]
    predictions = ["将来有了孩子，从幼儿园开始，就是无止境的择校费赞助费，临到高考还可能被遣回原籍……对异乡人而言，北京就像一个无边的巨大漩涡，你只有被裹挟着一圈圈打转，不知何时才能停下来喘口气"]
    config = {
        "output_dir": "results",
        "ignore_punct": True,
    }
    evaluator = Evaluator(config)
    metrics = evaluator.evaluate(sources, targets, predictions)
    print(f"{json.dumps(metrics, ensure_ascii=False)}")



if __name__ == "__main__":
# python -m evaluation.evaluator
    # main()
    test()
