# Stable Audio Open Prompting Experiments

该项目用于对比三种文本到环境音（ambience）生成条件在 Stable Audio Open 上的表现：

- `baseline`: 原始提示词直接生成。
- `structured`: 原始提示词先经 Qwen2-Audio 服务结构化，再生成。
- `cot`: 在 `structured` 基础上引入显式推理字段（`cot_trace` / `reasoning_tags`）后再生成。

项目会自动完成以下流程：

1. 读取提示词与实验配置。
2. 进行提示词编译（可选，取决于 condition）。
3. 调用 Stable Audio Open 生成音频。
4. 计算基础客观指标并输出 CSV 结果。

## 1. 项目结构

```text
project/
├── Qwen2AudioInstruct/
│   └── qwen2audio_server.py      # FastAPI 服务：/refine_prompt/schema, /refine_prompt/cot
├── prompts/
│   └── raw_prompts.json           # 原始提示词
├── scripts/
│   ├── run.sh                     # 一键运行（启动服务 + 跑实验）
│   ├── run_experiments.py         # 实验主流程
│   ├── qwen_prompt.py             # condition 分流与提示词转换
│   ├── generate_audio.py          # Stable Audio Open 生成封装
│   └── evaluate.py                # 指标计算与结果汇总
├── outputs/                       # 生成音频输出目录
├── metrics/
│   └── results.csv                # 结果表
├── config.yaml                    # 核心配置
└── requirements.txt               # Python 依赖
```

## 2. 环境准备

项目默认使用 conda 的 `dl` 环境运行（`scripts/run.sh` 已内置）：

```bash
conda create -n dl python=3.10 -y
conda activate dl
pip install --upgrade pip
pip install -r requirements.txt
```

说明：

- `run.sh` 会优先使用 `/data01/audio_group/d26_pengwenle/conda/bin/conda run -n dl`。
- 若找不到 conda，会退回系统 `python`。

## 3. 一键运行（推荐）

在项目根目录执行：

```bash
cd SAO/project
bash scripts/run.sh
```

脚本会执行：

1. 清理旧输出（`outputs/*`、`metrics/results.csv`）。
2. 启动 Qwen FastAPI 服务（`uvicorn Qwen2AudioInstruct.qwen2audio_server:app`）。
3. 检查服务健康（`http://127.0.0.1:8008/docs`）。
4. 运行 `scripts/run_experiments.py --config config.yaml`。
5. 结束后停止服务。

若服务超时未就绪，脚本会打印 `logs/qwen_server.log` 尾部日志并退出。

## 4. 分步运行

### 4.1 手动启动 Qwen 服务

```bash
cd SAO/project
conda run -n dl --no-capture-output python -m uvicorn Qwen2AudioInstruct.qwen2audio_server:app --host 127.0.0.1 --port 8008
```

### 4.2 手动运行实验

```bash
cd SAO/project
conda run -n dl --no-capture-output python scripts/run_experiments.py --config config.yaml
```

### 4.3 仅生成单条音频

```bash
cd SAO/project
conda run -n dl --no-capture-output python scripts/generate_audio.py \
  --prompt "forest ambience, birds chirping, light wind" \
  --output outputs/baseline/sample.wav \
  --seed 1000
```

## 5. 配置说明（config.yaml）

关键字段：

- `prompts_file`: 提示词 JSON 路径。
- `outputs_root`: 音频输出目录。
- `metrics_file`: 指标 CSV 输出路径。
- `sample_rate`, `duration_seconds`: 采样率与时长。
- `num_seeds`, `seed_offset`: 每个提示词的随机种子组。
- `model_name`: SAO 模型名（默认 `stabilityai/stable-audio-open-1.0`）。
- `sampler.*`: 扩散采样参数（steps、cfg、sigma、sampler_type）。
- `conditions.*.enabled`: 是否启用 `baseline/structured/cot` 条件。

## 6. 提示词数据格式

`prompts/raw_prompts.json` 示例：

```json
[
  {
    "id": "nature",
    "prompt": "forest ambience, birds chirping, light wind, leaves rustling, calm atmosphere"
  }
]
```

说明：

- `id` 用于输出文件命名和结果分组。
- 输出 wav 文件格式：`{condition}_{prompt_id}_{seed}.wav`。

## 7. 输出结果

- 音频输出：
  - `outputs/baseline/`
  - `outputs/structured/`
  - `outputs/cot/`
- 统计结果：`metrics/results.csv`

`results.csv` 包含：

- 每条音频的明细行（`row_type=per_audio`）。
- 按 `condition + prompt_id` 聚合的 summary 行（`row_type=prompt_summary`）。

## 8. 指标说明

当前实现包含：

- `loudness_variance`: RMS 响度方差。
- `spectral_centroid_variance`: 频谱质心方差。
- `embedding_similarity`: CLAP 文本-音频相似度（可选，依赖 `stable-audio-metrics`）。
- `fad`: Frechet Audio Distance（可选，需要背景参考音频目录）。

注意：指标用于辅助比较，不等价于主观听感质量。

## 9. 常见问题

### Q1: `run.sh` 卡在 Wait 阶段

排查步骤：

1. 查看 `logs/qwen_server.log`。
2. 检查端口占用：`lsof -i :8008`。
3. 确认 `dl` 环境中安装了 `fastapi/uvicorn/transformers/torch`。

### Q2: `structured` 或 `cot` 报服务连接错误

- 确认 Qwen 服务已启动。
- 检查 `QWEN_SERVICE_URL_BASE` 是否可访问。

### Q3: 模型下载或加载失败

- 检查网络与 Hugging Face 缓存路径权限。
- 可通过环境变量指定本地模型目录（如 `QWEN_LOCAL_MODEL_DIR`）。

## 10. 复现实验建议

- 固定 `num_seeds`、`seed_offset` 与 `sampler` 参数。
- 保留每次实验的 `config.yaml` 与 `prompts/raw_prompts.json` 快照。
- 对比不同条件时使用同一批 `prompt_id` 与 `seed`。
