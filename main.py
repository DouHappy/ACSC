#!/usr/bin/env python3
"""
大模型中文错别字纠正系统主程序入口

使用方法:
    python main.py --config config/config.yaml --mode full
    python main.py --config config/config.yaml --mode inference
    python main.py --config config/config.yaml --mode evaluate --results results/inference_results.json
"""

import argparse
import sys
import os

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from csc.csc_manager import CSCManager
from utils.logging_utils import setup_logging, get_logger


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="大模型中文错别字纠正系统",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 完整流程：推理+评估
  python main.py --config config/config.yaml --mode full
  
  # 仅推理
  python main.py --config config/config.yaml --mode inference
  
  # 仅评估已有结果
  python main.py --config config/config.yaml --mode evaluate --results results.jsonl
        """
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="配置文件路径 (默认: config/config.yaml)"
    )
    
    parser.add_argument(
        "--mode",
        type=str,
        choices=["full", "inference", "evaluate"],
        default="full",
        help="运行模式: full(完整流程), inference(仅推理), evaluate(仅评估)"
    )
    
    parser.add_argument(
        "--results",
        type=str,
        help="评估模式下使用的推理结果文件路径"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        help="输出目录 (覆盖配置文件中的设置)"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        help="批次大小 (覆盖配置文件中的设置)"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        help="模型名称或路径 (覆盖配置文件中的设置)"
    )
    
    parser.add_argument(
        "--dataset",
        type=str,
        help="数据集名称或路径 (覆盖配置文件中的设置)"
    )
    
    parser.add_argument(
        "--prompt",
        type=str,
        help="使用的prompt模板名称 (覆盖配置文件中的设置)"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="显示详细日志"
    )
    
    return parser.parse_args()


def validate_args(args):
    """验证参数有效性"""
    if not os.path.exists(args.config):
        print(f"错误: 配置文件不存在: {args.config}")
        return False
    
    if args.results and not os.path.exists(args.results):
        print(f"错误: 结果文件不存在: {args.results}")
        return False
    
    return True


def main():
    """主函数"""
    args = parse_args()
    
    # 验证参数
    if not validate_args(args):
        sys.exit(1)
    
    # 设置日志
    log_level = "DEBUG" if args.verbose else "INFO"
    setup_logging(log_level=log_level)
    logger = get_logger(__name__)
    
    logger.info("=" * 60)
    logger.info("大模型中文错别字纠正系统启动")
    logger.info("=" * 60)
    
    try:
        # 创建CSC管理器
        csc_manager = CSCManager(
            config_path=args.config
        )
        
        # 覆盖配置参数
        overrides = {}
        if args.output_dir:
            overrides['output_dir'] = args.output_dir
        if args.batch_size:
            overrides['pipeline']['batch_size'] = args.batch_size
        if args.model:
            overrides['model']['name'] = args.model
        if args.dataset:
            overrides['dataset']['name'] = args.dataset
        if args.prompt:
            overrides['prompts']['default_prompt'] = args.prompt
        
        if overrides:
            csc_manager.config.update(overrides)
            logger.info(f"已覆盖配置参数: {list(overrides.keys())}")
        
        # 根据模式运行
        if args.mode == "full":
            logger.info("运行模式: 完整流程 (推理 + 评估)")
            csc_manager.run_full_pipeline()
            
        elif args.mode == "inference":
            logger.info("运行模式: 仅推理")
            csc_manager.run_inference()
            
        elif args.mode == "evaluate":
            logger.info("运行模式: 仅评估")
            if not args.results:
                results_path = os.path.join(csc_manager.config['evaluation']['output_dir'], "results/inference_results.json")
            else:
                results_path = args.results
            
            if not os.path.exists(results_path):
                logger.error(f"结果文件不存在: {results_path}")
                if args.results:
                    print(f"请检查提供的results文件是否存在, {results_path}")
                else:
                    print(f"请检查默认的results文件是否存在, {results_path}")
                sys.exit(1)
            results = []
            
            import json
            with open(results_path, "r") as f:
                results = [json.loads(line) for line in f.readlines()]
            sources = [item['source'] for item in results]
            targets = [item['target'] for item in results]
            predictions = [item['prediction'] for item in results]
            
            from evaluation.evaluator import Evaluator
            csc_manager.evaluator = Evaluator(csc_manager.config['evaluation'])
            metrics = csc_manager.evaluator.evaluate(sources, targets, predictions)
            logger.info(f"评估指标: {metrics}")
        
        logger.info("任务完成！")
        
    except KeyboardInterrupt:
        logger.warning("用户中断任务")
        sys.exit(1)
    except Exception as e:
        logger.error(f"任务执行失败: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
