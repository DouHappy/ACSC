# Chinese Spelling Correction (CSC) Project

基于大语言模型的中文错别字纠正系统，支持多种LLM后端和评估指标计算。

## 项目结构

```
llmcsc/
├── config/           # 配置文件
├── data/             # 数据处理模块
├── prompts/          # Prompt模板管理
├── models/           # LLM模型管理
├── evaluation/       # 评估模块
├── pipeline/         # 任务执行流程
├── utils/            # 工具函数
├── README.md         # 项目说明
└── requirements.txt  # 依赖列表
```

## 功能特性

- 支持多种LLM后端（vLLM、Hugging Face）
- 模块化设计，易于扩展
- 支持In-context Learning (ICL)推理
- 字符级和句子级评估指标
- 断点恢复功能

## 安装依赖

```bash
pip install -r requirements.txt
```

## 使用方法

```bash
python run_csc.py --config config/default.yaml
