# 中文错别字纠正（CSC）

> **Chinese Spelling Correction powered by Large Language Models.**
该仓库提供了一套基于大模型（LLM）的中文错别字纠正解决方案，覆盖数据准备、Prompt 管理、批量推理、断点续跑、自动评估以及注意力可视化分析。借助统一的配置与命令行接口，用户可以在本地 GPU、Hugging Face Transformers 或 OpenAI 兼容 API 等多种推理后端之间灵活切换，快速验证与部署 CSC 能力。
## 目录

- [核心亮点](#核心亮点)
- [架构概览](#架构概览)
- [快速上手](#快速上手)
  - [环境准备](#环境准备)
  - [配置示例](#配置示例)
  - [运行模式](#运行模式)
- [数据与 Prompt 管理](#数据与-prompt-管理)
- [模型推理后端](#模型推理后端)
- [评估指标与输出](#评估指标与输出)
- [断点续跑与产物管理](#断点续跑与产物管理)
- [注意力统计与可视化](#注意力统计与可视化)
- [项目结构](#项目结构)
- [快速演示](#快速演示)
## 核心亮点
- **多后端推理**：统一调度 `vLLM`、Hugging Face Transformers 与 OpenAI 兼容 API，支持自定义 `base_url` 与生成参数，满足不同部署场景。
- **Prompt 灵活管理**
- **端到端流程控制**：`main.py` 提供完整流程（推理+评估）、仅推理、仅评估三种模式，并支持命令行覆盖配置项。
- **稳健的批处理能力**：推理过程中按批保存中间结果与检查点，异常情况下可快速恢复继续执行。
- **可解释性工具链**：附带注意力统计脚本与交互式可视化前端，帮助分析模型关注错别字的模式与位置。
## 架构概览
系统围绕 `CSCManager` 组织，采用模块化流水线：
1. **数据层**：`data/dataset_manager.py` 负责加载 `source \t target` 等格式的数据集，按配置生成批量样本。
2. **Prompt 层**：`prompts/prompt_manager.py` 读取 JSON 模板，拼接上下文并高亮错别字，对接不同类型模型输入。
3. **推理层**：`llm/llm_manager.py` 根据 `backend` 字段路由至对应实现，完成批量生成与必要后处理。
4. **评估层**：`evaluation/evaluator.py` 计算字符/句子级指标，并输出详细报告文件。
5. **产出管理**：`utils` 下的工具负责日志、文件系统、检查点与结果序列化。
通过 YAML 配置文件描述整个流水线，`csc/csc_manager.py` 根据配置串联各组件，实现高度模块化与可扩展的工作流。
## 快速上手
### 环境准备
```bash
# 方式一：使用 pip 安装基础依赖
pip install -r requirements.txt
# 方式二：使用 Conda 复现包含可视化的完整环境
conda env create -f environment.yml
conda activate csc
```
### 配置示例
在 `config/config.yaml` 填写数据、模型与推理参数。核心字段示例：
```yaml
dataset:
  path: "/path/to/your_csc.txt"     # 每行 [source]\t[target]
  format: "csc"
prompts:
  file_path: "prompts/prompts.json"
  name: "csc_icl"                    # 指定模板
  tokenizer_path: "/path/to/TokenizerOrModel"
model:
  model_name: "/path/to/ModelOrHFName"
  backend: "vllm"                    # 也可选 huggingface / api
  max_tokens: 512
  temperature: 0
  top_p: 0.9
  gpu_memory_utilization: 0.9
  tensor_parallel_size: 1
evaluation:
  output_dir: "results/your-run"
  ignore_punct: true                 # 可忽略中英标点
pipeline:
  type: "icl"
  batch_size: 4
  max_samples: null
  output_dir: "results/your-run"
logging:
  level: "INFO"
  file_path: "logs/csc.log"
  console: true
```
### 运行模式
```bash
# 完整流程：推理 + 评估
python main.py --config config/config.yaml --mode full
# 仅推理，可在命令行覆盖部分配置
python main.py --config config/config.yaml --mode inference \
  --batch-size 8 --model path/to/model --prompt csc_instruct
# 仅评估（可指定已有推理结果）
python main.py --config config/config.yaml --mode evaluate \
  --results path/to/inference_results.json
```
命令行解析与运行逻辑详见 `main.py`，覆盖参数会写入配置后再执行对应流程。
## 数据与 Prompt 管理
- **数据读取**：`DatasetManager` 支持按 `source\ttarget` 解析 CSC 数据，可通过 `format` 字段扩展其他格式，批处理由配置驱动。
- **Prompt 模板**：模板以 JSON 管理，支持 Base（单字符串）与 Instruct（消息列表）两种结构，必要时调用 Tokenizer 的 `apply_chat_template` 生成最终输入。
- **上下文增强**：配置中的 `pipeline.type` 可以选择 ICL 等策略，Prompt Manager 会自动拼接示例与高亮标注。
## 模型推理后端
`LLMManager` 统一封装模型调用接口，根据 `backend` 字段路由：
- **`vllm`**：面向大规模批处理与高吞吐推理，支持 GPU 资源配置。
- **`huggingface`**：直接使用 Transformers 的 CausalLM，实现快速本地测试。
- **`api`**：对接 OpenAI 兼容接口，支持自定义 `base_url` 与 `api_key`，适合云端服务。
推理结果默认包含 `prompt`、`prediction` 等字段，为后续评估和注意力分析提供原始信息。
## 评估指标与输出
- `Evaluator` 提供字符级、句子级 Precision/Recall/F1 及句子级 FPR 指标，可按需忽略中英标点。
- 评估结果默认写入 `evaluation.output_dir` 下的 `metrics.json`、`detailed_results.json` 等文件，便于复查与归档。
- 在仅评估模式下可通过 `--results` 指定自定义的推理结果文件。
## 断点续跑与产物管理
- 推理过程中会定期保存 `inference_results.json` 与检查点，记录已处理的样本索引及中间结果，异常恢复后可继续处理剩余数据。
- `utils/checkpoint_utils.py` 提供通用的加载、保存、合并逻辑，可按需扩展检查点内容。
- 所有产物默认存放在 `pipeline.output_dir`，与配置绑定便于管理。
## 注意力统计与可视化
项目附带了一套注意力分析工具链，帮助理解模型在错别字场景下的注意力分布：
1. **导出注意力张量**：`exp/statistic_attn.py` 从推理结果中读取 `prompt` 与 `prediction`，调用模型输出注意力矩阵，生成 `attention_batch_{start}_{end}.npz` 文件。
2. **安装可视化前端**：
   ```bash
   pip install -e attention-visualizer
   ```
3. **启动服务**：
   ```bash
   attention-viewer run \
     --data-dir /path/to/npz_dir \
     --tokenizer "/path/to/TokenizerOrHFName" \
     --host 0.0.0.0 \
     --port 8000
   ```
   前端基于 FastAPI + Plotly，可多层多头浏览、错别字高亮、Token 对齐。
更多细节可参考子项目文档 `attention-visualizer/README.md`。
## 项目结构
```
config/                        # 任务配置（YAML）
data/                          # 数据读取与格式化
llm/                           # LLM 后端与统一管理器
evaluation/                    # 评估指标与报告
prompts/                       # Prompt 管理与模板
utils/                         # 日志、文件、检查点等工具
attention-visualizer/          # 注意力可视化子项目（可独立安装）
exp/                           # 实验与分析脚本（注意力统计等）
csc/                           # 主流水线调度与业务逻辑
main.py                        # CLI 主入口
run_demo.py                    # 快速演示脚本（小样例）
requirements.txt               # pip 依赖列表
environment.yml                # Conda 环境（含可视化依赖）
```
## 快速演示
运行内置小样例体验完整流程：
```bash
python run_demo.py
```
待更新最新功能的demo
