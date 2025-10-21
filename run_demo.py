#!/usr/bin/env python3
"""
大模型中文错别字纠正系统 - 快速演示脚本

这个脚本提供了一个简单的演示，使用小数据集和轻量级模型进行测试
"""

import os
import json
import tempfile
from pathlib import Path

# 添加项目根目录到Python路径
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from csc.csc_manager import CSCManager
from utils.logging_utils import setup_logging, get_logger


def create_demo_config():
    """创建演示用的配置文件"""
    demo_config = {
        "dataset": {
            "name": "demo",
            "type": "custom",
            "path": "demo_data.txt",
            "split": "test",
            "max_samples": 10
        },
        "prompts": {
            "prompts_file": "prompts/prompts.json",
            "default_prompt": "base_correction"
        },
        "model": {
            "name": "THUDM/chatglm3-6b",
            "backend": "huggingface",
            "max_length": 512,
            "temperature": 0.1,
            "top_p": 0.9
        },
        "pipeline": {
            "batch_size": 2,
            "save_interval": 5,
            "max_workers": 1
        },
        "evaluation": {
            "metrics": ["char_precision", "char_recall", "char_f1", 
                       "sent_precision", "sent_recall", "sent_f1"],
            "output_format": "json"
        },
        "output_dir": "demo_output",
        "logging": {
            "level": "INFO",
            "file": "demo.log"
        }
    }
    
    # 保存配置文件
    with open("demo_config.yaml", "w", encoding="utf-8") as f:
        import yaml
        yaml.dump(demo_config, f, allow_unicode=True, indent=2)
    
    return "demo_config.yaml"


def create_demo_data():
    """创建演示用的测试数据"""
    demo_data = [
        "我今天很高兴，因为天气很好。\t我今天很高兴，因为天气很好。",
        "这个苹菓很好吃。\t这个苹果很好吃。",
        "他明天要去图收馆。\t他明天要去图书馆。",
        "我喜欢吃西红市。\t我喜欢吃西红柿。",
        "这个问题很难，我考虚了很久。\t这个问题很难，我考虑了很久。",
        "昨天下雨了，我忘记带伞了。\t昨天下雨了，我忘记带伞了。",
        "这个电题很好看。\t这个电影很好看。",
        "我要去超是买东西。\t我要去超市买东西。",
        "他的汉服很漂亮。\t他的汉服很漂亮。",
        "我想学习中文，但是很难。\t我想学习中文，但是很难。"
    ]
    
    with open("demo_data.txt", "w", encoding="utf-8") as f:
        for line in demo_data:
            f.write(line + "\n")
    
    return "demo_data.txt"


def run_demo():
    """运行演示"""
    print("🚀 启动大模型中文错别字纠正系统演示")
    print("=" * 50)
    
    # 设置日志
    setup_logging("INFO")
    logger = get_logger(__name__)
    
    try:
        # 创建演示数据
        logger.info("创建演示数据...")
        demo_data_path = create_demo_data()
        demo_config_path = create_demo_config()
        
        print(f"✅ 演示数据已创建: {demo_data_path}")
        print(f"✅ 演示配置已创建: {demo_config_path}")
        
        # 检查必要的依赖
        try:
            import torch
            print(f"✅ PyTorch版本: {torch.__version__}")
        except ImportError:
            print("❌ 未安装PyTorch，请先安装: pip install torch")
            return
        
        try:
            import transformers
            print(f"✅ Transformers版本: {transformers.__version__}")
        except ImportError:
            print("❌ 未安装Transformers，请先安装: pip install transformers")
            return
        
        # 创建CSC管理器
        logger.info("初始化系统...")
        csc_manager = CSCManager(config_path=demo_config_path)
        
        print("\n📊 演示数据预览:")
        with open(demo_data_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f, 1):
                if i <= 3:  # 只显示前3条
                    source, target = line.strip().split("\t")
                    print(f"  {i}. 原文: {source}")
                    print(f"     正确: {target}")
                    print()
        
        print("🎯 开始运行演示...")
        print("注意: 首次运行可能需要下载模型，请耐心等待")
        print("-" * 50)
        
        # 运行完整流程
        csc_manager.run_full_pipeline()
        
        print("-" * 50)
        print("✅ 演示完成！")
        print(f"📁 结果保存在: demo_output/")
        
        # 显示结果摘要
        results_file = "demo_output/results.jsonl"
        if os.path.exists(results_file):
            with open(results_file, "r", encoding="utf-8") as f:
                results = [json.loads(line) for line in f]
            
            print(f"\n📈 处理了 {len(results)} 条数据")
            
            # 显示前几条结果
            print("\n📝 部分结果预览:")
            for i, result in enumerate(results[:3], 1):
                print(f"  {i}. 原文: {result['source']}")
                print(f"     预测: {result['prediction']}")
                print(f"     正确: {result['target']}")
                print(f"     是否完全正确: {'✅' if result['is_correct'] else '❌'}")
                print()
        
        # 显示评估结果
        eval_file = "demo_output/evaluation_report.json"
        if os.path.exists(eval_file):
            with open(eval_file, "r", encoding="utf-8") as f:
                eval_results = json.load(f)
            
            print("📊 评估指标:")
            for metric, value in eval_results.items():
                if isinstance(value, float):
                    print(f"  {metric}: {value:.4f}")
                else:
                    print(f"  {metric}: {value}")
        
        print("\n🎉 演示结束！")
        print("您现在可以使用自己的数据和配置运行完整系统:")
        print("  python main.py --config config/config.yaml --mode full")
        
    except KeyboardInterrupt:
        print("\n\n⏹️  演示被用户中断")
    except Exception as e:
        print(f"\n❌ 演示运行失败: {str(e)}")
        logger.error("演示运行失败", exc_info=True)


if __name__ == "__main__":
    run_demo()
