"""
File utility functions for handling data and results.
"""

import os
import json
from typing import List, Dict, Any, Tuple
import logging
import yaml

logger = logging.getLogger(__name__)


def read_csc_data(file_path: str) -> List[Tuple[str, str]]:
    """
    读取CSC数据文件，格式为[source]\t[target]
    
    Args:
        file_path: 数据文件路径
        
    Returns:
        包含(source, target)元组的列表
    """
    data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                    
                parts = line.split('\t')
                if len(parts) != 2:
                    logger.warning(f"第{line_num}行格式错误: {line}")
                    continue
                    
                source, target = parts
                data.append((source, target))
                
    except FileNotFoundError:
        logger.error(f"文件未找到: {file_path}")
        raise
    except Exception as e:
        logger.error(f"读取文件时出错: {e}")
        raise
        
    return data


def save_results(results: List[Dict[str, Any]], output_path: str, append: bool = False):
    """
    保存推理结果到文件
    
    Args:
        results: 推理结果列表
        output_path: 输出文件路径
        append: 是否追加到现有文件
    """
    mode = 'a' if append else 'w'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    try:
        with open(output_path, mode, encoding='utf-8') as f:
            for result in results:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')
    except Exception as e:
        logger.error(f"保存结果时出错: {e}")
        raise


def load_results(result_path: str) -> List[Dict[str, Any]]:
    """
    从文件加载推理结果
    
    Args:
        result_path: 结果文件路径
        
    Returns:
        结果列表
    """
    results = []
    try:
        with open(result_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    results.append(json.loads(line))
    except FileNotFoundError:
        logger.warning(f"结果文件未找到: {result_path}")
    except Exception as e:
        logger.error(f"加载结果时出错: {e}")
        raise
        
    return results


def ensure_dir(file_path: str):
    """确保目录存在，如果不存在则创建"""
    directory = os.path.dirname(file_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)

def load_config(config_path: str) -> Dict[str, Any]:
    """
    加载任务配置

    Args:
        config_path: 配置文件yaml路径

    Returns:
        任务配置的字典
    """
    try:
        with open(config_path, "r", encoding="utf-8") as file:
            config = yaml.safe_load(file)
    except FileNotFoundError:
        logger.error(f"配置文件未找到: {config_path}")
        raise
    except Exception as e:
        logger.error(f"加载配置时出错: {e}")
        raise

    return config
