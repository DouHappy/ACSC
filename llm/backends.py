"""
LLM后端实现模块

提供vLLM和Huggingface Transformers的具体实现
"""

import abc
import logging
from typing import Dict, Any, List, Union, Optional
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from vllm import LLM, SamplingParams

from utils.logging_utils import get_logger

logger = get_logger(__name__)


class BaseBackend(abc.ABC):
    """LLM后端基类"""
    
    def __init__(self, config):
        self.config = config
        self.model = None
        self.tokenizer = None
        
    @abc.abstractmethod
    def generate(self, prompts: Union[str, List[str]], **kwargs) -> Union[str, List[str]]:
        """生成文本"""
        pass
    
    @abc.abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        pass
    
    @abc.abstractmethod
    def shutdown(self):
        """关闭模型，释放资源"""
        pass

class APIBackend(BaseBackend):
    """API后端实现，支持OpenAI、OpenRouter等提供商"""

    def __init__(self, config):
        super().__init__(config)
        self._initialize_client()
        
    def _initialize_client(self):
        """初始化API客户端"""
        try:
            import openai  # 假设已安装openai库，支持OpenAI和兼容API
            import os
            base_url = self.config.base_url
            
            if os.environ['API_KEY'] is None:
                raise ValueError("API_KEY environment variable not set")
            if os.environ['API_KEY'] == "":
                raise ValueError("API_KEY environment variable is empty")
            
            print(f"API_KEY={os.environ['API_KEY']}")
            self.client = openai.OpenAI(
                api_key=os.environ['API_KEY'],
                base_url=base_url,
            )
            self.model_name = self.config.model_name  # 如 'gpt-4' 或 OpenRouter模型ID
            
            logger.info(f"API backend initialized for base_url: {self.config.base_url}, model: {self.model_name}")
            
        except ImportError as e:
            logger.error("openai library not installed. Please install it via pip install openai")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize API client: {e}")
            raise
    
    def generate(self, prompts: Union[str, List[str]], **kwargs) -> Union[str, List[str]]:
        """使用API生成文本"""
        if isinstance(prompts, str):
            prompts = [prompts]
            return_single = True
        else:
            return_single = False
            
        results = []
        for prompt in prompts:
            try:
                response = self.client.completions.create(
                    model=self.model_name,
                    prompt=prompt,
                    max_tokens=kwargs.get("max_tokens", self.config.max_tokens),
                    temperature=kwargs.get("temperature", self.config.temperature),
                    top_p=kwargs.get("top_p", self.config.top_p),
                    # top_k 不直接支持，OpenAI使用n=1模拟
                )
                generated_text = response.choices[0].text.strip()
                results.append(generated_text)
            except Exception as e:
                logger.error(f"API generation failed for prompt: {prompt[:50]}...: {e}")
                raise
        
        return results[0] if return_single else results
    
    def chat_generate(self, messages: Union[Dict[str, str], List[Dict[str, str]]], **kwargs) -> Union[str, List[str]]:
        """对话式生成（API支持）"""
        if isinstance(messages, dict):
            messages = [messages]  # 假设是单组消息列表
            return_single = True
        else:
            return_single = False
            
        results = []
        for msg_list in messages:
            # 确保msg_list是List[Dict{'role': str, 'content': str}]
            if not isinstance(msg_list, list):
                msg_list = [msg_list]
            
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=msg_list,
                    max_tokens=kwargs.get("max_tokens", self.config.max_tokens),
                    temperature=kwargs.get("temperature", self.config.temperature),
                    top_p=kwargs.get("top_p", self.config.top_p),
                )
                generated_text = response.choices[0].message.content.strip()
                results.append(generated_text)
            except Exception as e:
                logger.error(f"API chat generation failed: {e}")
                raise
        
        return results[0] if return_single else results
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取API模型信息"""
        return {
            "backend": "api",
            "provider": self.provider,
            "model_name": self.model_name,
        }
    
    def shutdown(self):
        """关闭API后端"""
        self.client = None
        logger.info("API backend shutdown")

class VLLMBackend(BaseBackend):
    """vLLM后端实现"""
    
    def __init__(self, config):
        super().__init__(config)
        self._initialize_model()
        
    def _initialize_model(self):
        """初始化vLLM模型"""
        try:
            # 初始化vLLM
            self.model = LLM(
                model=self.config.model_name,
                tensor_parallel_size=self.config.tensor_parallel_size,
                gpu_memory_utilization=self.config.gpu_memory_utilization,
                trust_remote_code=self.config.trust_remote_code,
                max_model_len=self.config.max_model_len,
            )
            
            # 获取tokenizer
            self.tokenizer = self.model.get_tokenizer()
            
            logger.info(f"vLLM model loaded: {self.config.model_name}")
            
        except Exception as e:
            logger.error(f"Failed to initialize vLLM model: {e}")
            raise
    
    def generate(self, prompts: Union[str, List[str]], **kwargs) -> Union[str, List[str]]:
        """使用vLLM生成文本"""
        if isinstance(prompts, str):
            prompts = [prompts]
            return_single = True
        else:
            return_single = False
            
        # 设置采样参数
        sampling_params = SamplingParams(
            max_tokens=kwargs.get("max_tokens", self.config.max_tokens),
            temperature=kwargs.get("temperature", self.config.temperature),
            top_p=kwargs.get("top_p", self.config.top_p),
            top_k=kwargs.get("top_k", self.config.top_k),
        )
        
        try:
            # 生成文本
            outputs = self.model.generate(prompts, sampling_params)
            
            # 提取生成的文本
            results = [output.outputs[0].text.strip() for output in outputs]
            
            return results[0] if return_single else results
            
        except Exception as e:
            logger.error(f"vLLM generation failed: {e}")
            raise
    
    def chat_generate(self, messages: Union[Dict[str, str], List[Dict[str, str]]], **kwargs) -> Union[str, List[str]]:
        """对话式生成（vLLM支持）"""
        if isinstance(messages, dict):
            messages = [messages]
            return_single = True
        else:
            return_single = False
            
        # 将messages格式化为prompt
        if hasattr(self.tokenizer, 'apply_chat_template'):
            prompts = []
            for msg_list in messages:
                prompt = self.tokenizer.apply_chat_template(
                    msg_list,
                    tokenize=False,
                    add_generation_prompt=True
                )
                prompts.append(prompt)
        else:
            # 简单的格式化方式
            prompts = []
            for msg_list in messages:
                prompt = ""
                for msg in msg_list:
                    role = msg.get("role", "user")
                    content = msg.get("content", "")
                    prompt += f"{role}: {content}\n"
                prompt += "assistant: "
                prompts.append(prompt)
        
        return self.generate(prompts, **kwargs)
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取vLLM模型信息"""
        if self.model is None:
            return {}
            
        return {
            "backend": "vllm",
            "model_name": self.config.model_name,
            "tensor_parallel_size": self.config.tensor_parallel_size,
            "gpu_memory_utilization": self.config.gpu_memory_utilization,
            "max_model_len": self.model.llm_engine.model_config.max_model_len,
            "tokenizer_vocab_size": len(self.tokenizer) if self.tokenizer else None,
        }
    
    def shutdown(self):
        """关闭vLLM模型"""
        if self.model is not None:
            # vLLM会自动清理资源
            self.model = None
            self.tokenizer = None
            logger.info("vLLM backend shutdown")


class HuggingFaceBackend(BaseBackend):
    """Huggingface Transformers后端实现"""
    
    def __init__(self, config):
        super().__init__(config)
        self._initialize_model()
        
    def _initialize_model(self):
        """初始化Huggingface模型"""
        try:
            # 加载tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_name,
                trust_remote_code=self.config.trust_remote_code,
                **self.config.model_kwargs
            )
            
            # 设置pad_token
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
            # 加载模型
            device_map = "auto" if self.config.device == "auto" else self.config.device
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name,
                torch_dtype=torch.float16,
                device_map=device_map,
                trust_remote_code=self.config.trust_remote_code,
                **self.config.model_kwargs
            )
            
            logger.info(f"Huggingface model loaded: {self.config.model_name}")
            
        except Exception as e:
            logger.error(f"Failed to initialize Huggingface model: {e}")
            raise
    
    def generate(self, prompts: Union[str, List[str]], **kwargs) -> Union[str, List[str]]:
        """使用Huggingface Transformers生成文本"""
        if isinstance(prompts, str):
            prompts = [prompts]
            return_single = True
        else:
            return_single = False
            
        # 编码输入
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.tokenizer.model_max_length
        )
        
        # 移动到GPU
        if torch.cuda.is_available():
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        # 设置生成配置
        generation_config = GenerationConfig(
            max_new_tokens=kwargs.get("max_tokens", self.config.max_tokens),
            temperature=kwargs.get("temperature", self.config.temperature),
            top_p=kwargs.get("top_p", self.config.top_p),
            top_k=kwargs.get("top_k", self.config.top_k),
            do_sample=True,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        
        try:
            # 生成文本
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    generation_config=generation_config
                )
            
            # 解码输出
            results = []
            for i, (prompt, output) in enumerate(zip(prompts, outputs)):
                # 移除输入部分
                generated_text = self.tokenizer.decode(
                    output[inputs["input_ids"].shape[1]:],
                    skip_special_tokens=True
                ).strip()
                results.append(generated_text)
            
            return results[0] if return_single else results
            
        except Exception as e:
            logger.error(f"Huggingface generation failed: {e}")
            raise
    
    def chat_generate(self, messages: Union[Dict[str, str], List[Dict[str, str]]], **kwargs) -> Union[str, List[str]]:
        """对话式生成（Huggingface支持）"""
        if isinstance(messages, dict):
            messages = [messages]
            return_single = True
        else:
            return_single = False
            
        # 将messages格式化为prompt
        if hasattr(self.tokenizer, 'apply_chat_template'):
            prompts = []
            for msg_list in messages:
                prompt = self.tokenizer.apply_chat_template(
                    msg_list,
                    tokenize=False,
                    add_generation_prompt=True
                )
                prompts.append(prompt)
        else:
            # 简单的格式化方式
            prompts = []
            for msg_list in messages:
                prompt = ""
                for msg in msg_list:
                    role = msg.get("role", "user")
                    content = msg.get("content", "")
                    prompt += f"{role}: {content}\n"
                prompt += "assistant: "
                prompts.append(prompt)
        
        return self.generate(prompts, **kwargs)
    
    def get_model_info(self) -> Dict[str, Any]:
        """获取Huggingface模型信息"""
        if self.model is None:
            return {}
            
        return {
            "backend": "huggingface",
            "model_name": self.config.model_name,
            "device": str(next(self.model.parameters()).device),
            "dtype": str(next(self.model.parameters()).dtype),
            "num_parameters": sum(p.numel() for p in self.model.parameters()),
            "tokenizer_vocab_size": len(self.tokenizer) if self.tokenizer else None,
        }
    
    def shutdown(self):
        """关闭Huggingface模型"""
        if self.model is not None:
            # 清理GPU内存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            self.model = None
            self.tokenizer = None
            logger.info("Huggingface backend shutdown")
