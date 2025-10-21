"""LLM管理模块
提供统一的LLM接口，支持多种后端引擎
"""

from .llm_manager import LLMManager
from .backends import VLLMBackend, HuggingFaceBackend

__all__ = ["LLMManager", "VLLMBackend", "HuggingFaceBackend"]
