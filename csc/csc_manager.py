"""
CSCManager - 中文拼写纠错任务管理器
负责协调整个CSC任务的执行流程
"""

import json
import logging
from typing import Dict, List, Any, Optional
from pathlib import Path
import time
from datetime import datetime
from tqdm import tqdm

from data.dataset_manager import DatasetManager
from prompts.prompt_manager import PromptManager
from llm.llm_manager import LLMManager
from evaluation.evaluator import Evaluator
from utils.file_utils import load_config, save_results, ensure_dir
from utils.logging_utils import get_logger, setup_logging
from utils.checkpoint_utils import CheckpointManager, ResumeManager

logger = get_logger(__name__)


class CSCManager:
    """中文拼写纠错任务管理器"""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        初始化CSC管理器
        
        Args:
            config_path: 配置文件路径
        """
        self.config = load_config(config_path)
        self.setup_environment()
        
        # 初始化各个管理器
        self.dataset_manager = None
        self.prompt_manager = None
        self.llm_manager = None
        self.evaluator = None
        
        # 检查点管理
        self.checkpoint_manager = None
        self.resume_manager = None
        
        # 任务状态
        self.task_status = {
            'initialized': False,
            'data_loaded': False,
            'prompts_loaded': False,
            'model_loaded': False,
            'inference_started': False,
            'inference_completed': False,
            'evaluation_completed': False
        }
        
        logger.info("CSCManager初始化完成")
    
    def setup_environment(self):
        """设置运行环境"""
        # 设置日志
        log_config = self.config.get('logging', {})
        setup_logging(
            log_level=log_config.get('level', 'INFO'),
            log_file=log_config.get('file', 'logs/csc.log'),
            log_format=log_config.get('format')
        )
        
        # 创建必要的目录
        output_dir = Path(self.config.get('pipeline', {}).get('output_dir', 'outputs'))
        ensure_dir(output_dir)
        ensure_dir(output_dir / 'checkpoints')
        ensure_dir(output_dir / 'results')
        ensure_dir(output_dir / 'logs')
        
        logger.info("运行环境设置完成")
    
    def initialize_components(self):
        """初始化各个组件"""
        try:
            # 初始化数据集管理器
            self.dataset_manager = DatasetManager(self.config['dataset'])
            logger.info("DatasetManager初始化完成")
            
            # 初始化提示管理器
            self.prompt_manager = PromptManager(self.config['prompts']['file_path'], self.config['prompts']['tokenizer_path'])
            logger.info("PromptManager初始化完成")
            
            # 初始化LLM管理器
            self.llm_manager = LLMManager(self.config['model'])
            logger.info("LLMManager初始化完成")
            # 初始化评估器
            self.evaluator = Evaluator(self.config['evaluation'])
            logger.info("Evaluator初始化完成")
            
            # 初始化检查点管理器
            checkpoint_dir = Path(self.config.get('pipeline', {}).get('output_dir', 'outputs')) / 'checkpoints'
            self.checkpoint_manager = CheckpointManager(str(checkpoint_dir))
            self.resume_manager = ResumeManager(str(checkpoint_dir))
            
            self.task_status['initialized'] = True
            logger.info("所有组件初始化完成")
            
        except Exception as e:
            logger.error(f"组件初始化失败: {e}")
            raise
    
    def load_data(self) -> List[Dict[str, str]]:
        """加载数据集"""
        if not self.task_status['initialized']:
            self.initialize_components()
        
        try:
            # 加载数据
            data = self.dataset_manager.load_dataset()
            logger.info(f"成功加载 {len(data)} 条数据")
            
            self.task_status['data_loaded'] = True
            return data
            
        except Exception as e:
            logger.error(f"数据加载失败: {e}")
            raise
    
    @staticmethod
    def string_marker(s1, s2):
        """
        优化版本：合并连续的操作以获得更好的标记效果
        """
        m, n = len(s1), len(s2)
        
        # 创建DP表格
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        # 初始化边界条件
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j
        
        # 填充DP表格
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if s1[i-1] == s2[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = 1 + min(
                        dp[i-1][j],      # 删除
                        dp[i][j-1],      # 插入
                        dp[i-1][j-1]     # 替换
                    )
        
        # 回溯找到编辑路径
        operations = []
        i, j = m, n
        
        while i > 0 or j > 0:
            if i > 0 and j > 0 and s1[i-1] == s2[j-1]:
                operations.append(('match', s1[i-1]))
                i -= 1
                j -= 1
            elif i > 0 and j > 0 and dp[i][j] == dp[i-1][j-1] + 1:
                operations.append(('replace', s1[i-1], s2[j-1]))
                i -= 1
                j -= 1
            elif i > 0 and dp[i][j] == dp[i-1][j] + 1:
                operations.append(('delete', s1[i-1]))
                i -= 1
            elif j > 0 and dp[i][j] == dp[i][j-1] + 1:
                operations.append(('insert', s2[j-1]))
                j -= 1
        
        operations.reverse()
        
        # 合并连续的操作并构建结果
        result = ""
        i = 0
        
        while i < len(operations):
            op = operations[i]
            
            if op[0] == 'match':
                result += op[1]
                i += 1
            elif op[0] == 'insert':
                # 收集连续的插入操作
                insert_chars = ""
                while i < len(operations) and operations[i][0] == 'insert':
                    insert_chars += operations[i][1]
                    i += 1
                result += "<-->"
            elif op[0] == 'delete':
                # 收集连续的删除操作
                delete_chars = ""
                while i < len(operations) and operations[i][0] == 'delete':
                    delete_chars += operations[i][1]
                    i += 1
                result += f"<-{delete_chars}->"
            elif op[0] == 'replace':
                # 收集连续的替换操作
                replace_chars = ""
                while i < len(operations) and operations[i][0] == 'replace':
                    replace_chars += operations[i][1]
                    i += 1
                result += f"<-{replace_chars}->"
        
        return result
        
    def prepare_prompts(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """准备Prompt"""
        if not self.task_status['data_loaded']:
            data = self.load_data()
        
        try:
            prompt_name = self.config['prompts'].get('name', None)
            
            prompts = []
            for item in data:
                # 格式化提示
                split_char = self.config['pipeline'].get('split_token', False)
                source = item['source']
                target = item['target']
                if split_char:
                    source = split_char.join(item['source']) if split_char else item['source']
                    target = split_char.join(item['target']) if split_char else item['target']
                mark = self.config['pipeline'].get('mark', False)
                if mark:
                    source = self.string_marker(source, target)
                formatted_prompt = self.prompt_manager.format_prompt(
                    prompt_name,
                    source=source,
                    target=target,
                )
                prompts.append({
                    'source': item['source'],
                    'target': item['target'],
                    'prompt': formatted_prompt,
                })
            
            logger.info(f"成功准备 {len(prompts)} 个提示")
            self.task_status['prompts_loaded'] = True
            return prompts
            
        except Exception as e:
            logger.error(f"提示准备失败: {e}")
            raise
    
    def check_resume(self) -> Optional[Dict[str, Any]]:
        """检查是否可以恢复任务"""
        try:
            resume_info = self.resume_manager.check_resume()
            if resume_info:
                logger.info("检测到可恢复的任务")
                logger.info(f"上次处理进度: {resume_info.get('processed', 0)}/{resume_info.get('total', 0)}")
                return resume_info
            return None
            
        except Exception as e:
            logger.warning(f"检查恢复状态失败: {e}")
            return None
    
    def run_inference(self, prompts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """执行推理"""
        if not self.task_status['prompts_loaded']:
            prompts = self.prepare_prompts(self.load_data())
        
        try:
            # 检查是否可以恢复
            resume_info = self.check_resume()
            start_index = 0
            
            if resume_info:
                start_index = resume_info.get('processed', 0)
                logger.info(f"从第 {start_index} 条数据开始恢复")
            
            # 获取批次配置
            batch_config = self.config['pipeline'].get('batch', {})
            batch_size = batch_config.get('size', 10240)
            save_interval = batch_config.get('save_interval', 100)
            
            # 准备推理数据
            inference_prompts = prompts[start_index:]
            total_batches = (len(inference_prompts) + batch_size - 1) // batch_size
            
            logger.info(f"开始推理，共 {len(inference_prompts)} 条数据，{total_batches} 个批次")
            
            results = []
            if resume_info:
                # 加载之前的结果
                results = load_results(resume_info.get('result_file', ''))
            
            self.task_status['inference_started'] = True
            
            # 批次推理
            for batch_idx in tqdm(range(0, len(inference_prompts), batch_size), desc="Inference", leave=False):
                batch = inference_prompts[batch_idx:batch_idx + batch_size]
                
                # 提取提示
                batch_prompts = [item['prompt'] for item in batch]
                
                # 执行推理
                if isinstance(prompts[0], dict):
                    predictions = self.llm_manager.chat_generate(batch_prompts)
                else:
                    predictions = self.llm_manager.generate(batch_prompts)
                
                # 构建结果
                for item, pred in zip(batch, predictions):
                    result = {
                        'source': item['source'],
                        'target': item['target'],
                        'prediction': pred,
                        'prompt': item['prompt'],
                        'timestamp': datetime.now().isoformat()
                    }
                    results.append(result)
                
                # 保存检查点
                current_processed = start_index + batch_idx + len(batch)
                self.checkpoint_manager.save_checkpoint(
                    task_name=self.config.get('task_name', "csc"),
                    processed_index=current_processed,
                    metadata = {
                        "total": len(prompts),
                        "result": results,
                    }
                )
                
                # 定期保存结果
                if (batch_idx // batch_size + 1) % save_interval == 0:
                    self.save_intermediate_results(results)
                
                logger.info(f"处理进度: {current_processed}/{len(prompts)} "
                          f"({current_processed/len(prompts)*100:.1f}%)")
            
            # 保存最终结果
            final_result_file = self.save_final_results(results)
            
            # 更新评估配置
            self.config['evaluation']['result_file'] = final_result_file
            
            self.task_status['inference_completed'] = True
            logger.info("推理完成")
            
            return results
            
        except Exception as e:
            logger.error(f"推理失败: {e}")
            # 保存当前进度
            if 'results' in locals():
                self.save_intermediate_results(results)
            raise
    
    def save_intermediate_results(self, results: List[Dict[str, Any]]):
        """保存中间结果"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = Path(self.config['pipeline']['output_dir']) / 'results'
            intermediate_file = output_dir / f"intermediate_{timestamp}.json"
            
            save_results(results, str(intermediate_file))
            logger.info(f"中间结果已保存: {intermediate_file}")
            
        except Exception as e:
            logger.warning(f"保存中间结果失败: {e}")
    
    def save_final_results(self, results: List[Dict[str, Any]]) -> str:
        """保存最终结果"""
        try:
            output_dir = Path(self.config['pipeline']['output_dir']) / 'results'
            final_file = output_dir / "inference_results.json"
            
            save_results(results, str(final_file))
            logger.info(f"最终结果已保存: {final_file}")
            
            return str(final_file)
            
        except Exception as e:
            logger.error(f"保存最终结果失败: {e}")
            raise
    
    def run_evaluation(self) -> Dict[str, Any]:
        """运行评估"""
        if not self.task_status['inference_completed']:
            logger.warning("推理尚未完成，请先运行推理")
            return {}
        
        try:
            logger.info("开始评估")
            metrics = self.evaluator.evaluate()
            
            self.task_status['evaluation_completed'] = True
            logger.info("评估完成")
            
            return metrics
            
        except Exception as e:
            logger.error(f"评估失败: {e}")
            raise
    
    def run_full_pipeline(self) -> Dict[str, Any]:
        """运行完整流程"""
        try:
            logger.info("开始运行完整CSC流程")
            
            # 1. 初始化组件
            self.initialize_components()
            
            # 2. 加载数据
            data = self.load_data()
            
            # 3. 准备提示
            prompts = self.prepare_prompts(data)
            # 4. 执行推理
            results = self.run_inference(prompts)
            
            # 5. 运行评估
            metrics = self.run_evaluation()
            
            # 6. 生成报告
            report = self.generate_report(results, metrics)
            
            logger.info("CSC流程完成")
            return report
            
        except Exception as e:
            logger.error(f"流程执行失败: {e}")
            raise
    
    def generate_report(self, results: List[Dict[str, Any]], metrics: Dict[str, Any]) -> Dict[str, Any]:
        """生成任务报告"""
        try:
            report = {
                'task_info': {
                    'start_time': datetime.now().isoformat(),
                    'config': self.config,
                    'status': self.task_status
                },
                'data_info': {
                    'total_samples': len(results),
                    'dataset_config': self.config['dataset']
                },
                'model_info': {
                    'model_config': self.config['model'],
                    'prompt_config': self.config['pipeline'].get('prompt', {})
                },
                'results': {
                    'inference_results': len(results),
                    'evaluation_metrics': metrics
                }
            }
            
            # 保存报告
            report_file = Path(self.config['pipeline']['output_dir']) / 'results' / 'task_report.json'
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
            
            logger.info(f"任务报告已保存: {report_file}")
            return report
            
        except Exception as e:
            logger.error(f"生成报告失败: {e}")
            raise
    
    def get_status(self) -> Dict[str, Any]:
        """获取任务状态"""
        return {
            'status': self.task_status,
            'config': self.config,
            'timestamp': datetime.now().isoformat()
        }
    
    def cleanup(self):
        """清理资源"""
        try:
            if self.llm_manager:
                self.llm_manager.cleanup()
            
            logger.info("资源清理完成")
            
        except Exception as e:
            logger.warning(f"清理资源时出错: {e}")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='中文拼写纠错任务管理器')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                        help='配置文件路径')
    parser.add_argument('--mode', type=str, choices=['inference', 'evaluate', 'full'],
                        default='full', help='运行模式')
    parser.add_argument('--resume', action='store_true',
                        help='从检查点恢复任务')
    
    args = parser.parse_args()
    
    # 创建管理器
    manager = CSCManager(args.config)
    
    try:
        if args.mode == 'inference':
            # 仅运行推理
            data = manager.load_data()
            prompts = manager.prepare_prompts(data)
            results = manager.run_inference(prompts)
            
        elif args.mode == 'evaluate':
            # 仅运行评估
            metrics = manager.run_evaluation()
            print(json.dumps(metrics, ensure_ascii=False, indent=2))
            
        else:
            # 运行完整流程
            report = manager.run_full_pipeline()
            print(json.dumps(report, ensure_ascii=False, indent=2))
    
    finally:
        manager.cleanup()


if __name__ == "__main__":
    main()
