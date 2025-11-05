"""
LLM Manager Module

提供统一的LLM管理接口，支持多种后端引擎（vLLM、Huggingface等）
"""

import os
import logging
from typing import Dict, Any, List, Union, Optional
from dataclasses import dataclass, field

from .backends import VLLMBackend, HuggingFaceBackend, BaseBackend, APIBackend, AliyunDashScopeBackend
from utils.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class LLMConfig:
    """LLM配置数据类"""
    model_name: str
    backend: str = "vllm"  # vllm, huggingface
    max_tokens: int = 512
    batch_size: int = 8
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    trust_remote_code: bool = False
    temperature: float = 0.6
    top_p: float = 0.9
    top_k: int = 50
    device: str = "auto"
    max_model_len: int = 2048
    base_url: str = ""
    api_key: Optional[str] = None
    api_key_env: Optional[str] = None
    extra_body: Optional[Dict[str, Any]] = None
        
class LLMManager:
    """统一的LLM管理器，支持多种后端"""
    
    def __init__(self, config: Union[Dict[str, Any], LLMConfig]):
        """
        初始化LLM管理器
        
        Args:
            config: LLM配置，可以是字典或LLMConfig对象
        """
        if isinstance(config, dict):
            self.config = LLMConfig(**config)
        else:
            self.config = config
            
        self.backend: Optional[BaseBackend] = None
        self._initialize_backend()
        
    def _initialize_backend(self):
        """根据配置初始化后端"""
        backend_name = self.config.backend.lower()
        
        if backend_name == "vllm":
            self.backend = VLLMBackend(self.config)
        elif backend_name == "huggingface":
            self.backend = HuggingFaceBackend(self.config)
        elif backend_name == 'api':
            self.backend = APIBackend(self.config)
        elif backend_name in {'aliyuncs', 'dashscope'}:
            self.backend = AliyunDashScopeBackend(self.config)
        else:
            raise ValueError(f"Unsupported backend: {backend_name}")
            
        logger.info(f"Initialized {backend_name} backend with model: {self.config.model_name}")
        
    def generate(self, 
                prompts: Union[str, List[str]], 
                **kwargs) -> Union[str, List[str]]:
        """
        生成文本
        
        Args:
            prompts: 输入提示，可以是单个字符串或字符串列表
            **kwargs: 额外的生成参数，会覆盖config中的设置
            
        Returns:
            生成的文本，与输入格式对应
        """
        if self.backend is None:
            raise RuntimeError("Backend not initialized")
            
        # 合并配置和额外参数
        generation_config = {
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
            "temperature": kwargs.get("temperature", self.config.temperature),
            "top_p": kwargs.get("top_p", self.config.top_p),
            "top_k": kwargs.get("top_k", self.config.top_k),
        }
        generation_config.update(kwargs)
        
        return self.backend.generate(prompts, **generation_config)
    
    def generate_batch(self, 
                      prompts: List[str], 
                      batch_size: Optional[int] = None,
                      **kwargs) -> List[str]:
        """
        批量生成文本
        
        Args:
            prompts: 输入提示列表
            batch_size: 批次大小，如果为None则使用config中的设置
            **kwargs: 额外的生成参数
            
        Returns:
            生成的文本列表
        """
        if batch_size is None:
            batch_size = self.config.batch_size
            
        results = []
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i:i + batch_size]
            batch_results = self.generate(batch, **kwargs)
            if isinstance(batch_results, str):
                batch_results = [batch_results]
            results.extend(batch_results)
            
            logger.debug(f"Processed batch {i//batch_size + 1}/{(len(prompts) + batch_size - 1)//batch_size}")
            
        return results
    
    def chat_generate(self, 
                     messages: Union[Dict[str, str], List[Dict[str, str]]],
                     **kwargs) -> Union[str, List[str]]:
        """
        对话式生成（适用于Instruct模型）
        
        Args:
            messages: 消息列表，每个消息包含role和content
            **kwargs: 额外的生成参数
            
        Returns:
            生成的回复
        """
        if self.backend is None:
            raise RuntimeError("Backend not initialized")
            
        if not hasattr(self.backend, 'chat_generate'):
            raise NotImplementedError(f"Backend {self.config.backend} does not support chat generation")
            
        return self.backend.chat_generate(messages, **kwargs)
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        if self.backend is None:
            return {}
        return self.backend.get_model_info()
    
    def shutdown(self):
        """关闭后端，释放资源"""
        if self.backend is not None:
            self.backend.shutdown()
            logger.info("LLM backend shutdown completed")
            
    def __enter__(self):
        """上下文管理器入口"""
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.shutdown()

def main():
    config = LLMConfig(
        model_name="/data/images/llms/Qwen/Qwen2.5-1.5B-Instruct",
        backend="vllm",
        max_tokens=512,
        batch_size=8,
        gpu_memory_utilization=0.4,
        tensor_parallel_size=1,
        trust_remote_code=True,
        device="auto",
    )
    llm_manager = LLMManager(config)
    print(llm_manager.get_model_info())
    prompt = "你好"
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    chat_prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True
    )
    print(chat_prompt)
    print(llm_manager.generate(chat_prompt))

if __name__ == '__main__':
    main()
