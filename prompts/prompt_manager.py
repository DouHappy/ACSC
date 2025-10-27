"""
Prompt管理器模块
负责加载和管理prompt模板，支持base model和Instruct model的不同格式
"""

import json
import os
from copy import deepcopy
from transformers import AutoTokenizer
from typing import Dict, Any, List, Union
from dataclasses import dataclass
from jinja2 import Template, Environment, BaseLoader
import logging

logger = logging.getLogger(__name__)


@dataclass
class PromptTemplate:
    """Prompt模板数据结构"""
    name: str
    description: str
    template: str
    template_type: str = "base"  # "base" 或 "instruct"


class PromptManager:
    """Prompt管理器，负责加载和管理prompt模板"""
    
    def __init__(self, prompts_file: str = None, tokenizer_path: str = None, examples: List[Dict[str, str]] = None):
        """
        初始化PromptManager
        
        Args:
            prompts_file: prompt配置文件路径，默认为项目中的prompts.json
        """
        if prompts_file is None:
            # 使用项目默认的prompts.json
            current_dir = os.path.dirname(os.path.abspath(__file__))
            prompts_file = os.path.join(current_dir, "prompts.json")
        
        self.prompts_file = prompts_file
        self.prompts: Dict[str, PromptTemplate] = {}
        self._load_prompts()
        if tokenizer_path:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
        else:
            self.tokenizer = None
        self.examples = examples
    
    def _load_prompts(self) -> None:
        """从JSON文件加载prompt模板"""
        try:
            with open(self.prompts_file, 'r', encoding='utf-8') as f:
                prompts_data = json.load(f)
            
            for prompt_data in prompts_data:
                assert(isinstance(prompt_data["template"], str) or isinstance(prompt_data["template"], list))
                prompt = PromptTemplate(
                    name=prompt_data["name"],
                    description=prompt_data["description"],
                    template=prompt_data["template"],
                    template_type="instruct" if isinstance(prompt_data["template"], list) else "base"
                )
                self.prompts[prompt.name] = prompt
                logger.info(f"已加载prompt模板: {prompt.name}")
                
        except FileNotFoundError:
            logger.error(f"Prompt文件未找到: {self.prompts_file}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"JSON解析错误: {e}")
            raise
        except KeyError as e:
            logger.error(f"Prompt配置缺少必要字段: {e}")
            raise
    
    def get_prompt_template(self, name: str) -> PromptTemplate:
        """
        根据名称获取prompt模板
        
        Args:
            name: prompt模板名称
            
        Returns:
            PromptTemplate对象
            
        Raises:
            KeyError: 如果找不到指定名称的prompt
        """
        if name not in self.prompts:
            available = list(self.prompts.keys())
            raise KeyError(f"Prompt '{name}' 未找到。可用prompt: {available}")
        
        return self.prompts[name]
    
    def list_prompts(self) -> List[str]:
        """获取所有可用的prompt名称"""
        return list(self.prompts.keys())
    
    def get_prompt_info(self, name: str) -> Dict[str, str]:
        """获取prompt的详细信息"""
        prompt = self.get_prompt_template(name)
        return {
            "name": prompt.name,
            "description": prompt.description,
            "template_type": prompt.template_type
        }
    
    def get_few_shot_examples(self, few_shot: int = 0) -> str:
        """
        获取few-shot examples
        
        Args:
            few_shot: few-shot examples数量
            
        Returns:
            few-shot examples字符串
        """
        if few_shot >= len(self.examples):
            return random.sample(self.examples, few_shot)
        else:
            logger.error(f"few-shot examples数量({few_shot})大于总examples数量({len(self.examples)}), 只采样")
        pass

    def format_prompt(self, prompt_name: str, few_shot: int = 0, **kwargs) -> Union[str, List[Dict[str, str]]]:
        """
        根据模板和参数格式化prompt
        
        Args:
            prompt_name: prompt模板名称
            **kwargs: 模板参数
            
        Returns:
            格式化后的prompt
            - base model: 返回字符串
            - instruct model: 返回messages列表
            
        Raises:
            KeyError: 如果找不到指定名称的prompt
            ValueError: 如果模板参数不完整
        """
        prompt = deepcopy(self.get_prompt_template(prompt_name))
        if few_shot > 0:
            kwargs["examples"] = self.get_few_shot_examples(few_shot)
        try:
            if prompt.template_type == 'base':
                return prompt.template.format(**kwargs)
            else:
                for p in prompt.template:
                    p['content'] = p['content'].format(**kwargs)

                return prompt.template

        except Exception as e:
            logger.error(f"格式化prompt失败: {e}")
            raise ValueError(f"格式化prompt失败: {e}")

    def get_instruct_len(self, name: str) -> int:
        """
        计算Prompt Template 中 Instruction 的token长度
        
        Args:
            name: prompt模板名称
            tokenizer: 用于计算token长度的分词器
            
        Returns:
            token长度
        """
        test_source = "<test>This is a test source.</test>"
        prompt_message = self.format_prompt(name, source=test_source)
        prompt = self.tokenizer.apply_chat_template(prompt_message, tokenize=False)
        insturct = prompt[:prompt.find(test_source)]
        instruct_token_str = [self.tokenizer.decode(t) for t in self.tokenizer.encode(insturct)]
        return len(instruct_token_str)

    def get_token_str_list(self, text: str) -> List[str]:
        """
        将文本转换为token字符串列表
        
        Args:
            text: 输入文本
            
        Returns:
            token字符串列表
        """
        if not self.tokenizer:
            raise ValueError("Tokenizer未初始化，无法进行分词操作。")
        
        token_ids = self.tokenizer.encode(text)
        token_strs = [self.tokenizer.decode(tid) for tid in token_ids]
        return token_strs

# 使用示例
if __name__ == "__main__":
    # 初始化PromptManager
    pm = PromptManager()
    
    # 列出所有prompt
    print("可用prompt:", pm.list_prompts())
    
    # 格式化base model的prompt
    base_prompt = pm.format_prompt(
        "base",
        user_input="我今天很开新"
    )
    print("Base prompt:", base_prompt)
    
    # 格式化instruct model的prompt
    instruct_prompt = pm.format_prompt(
        "csc_icl_correction",
        user_input="我今天很开新",
        examples="原句: 我很高心见到你\n纠正: 我很高兴见到你"
    )
    print("Instruct prompt:", instruct_prompt)
