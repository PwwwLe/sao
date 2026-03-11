# SAO 一键启动说明

本项目提供统一脚本用于启动、停止、查看状态和查看日志。

脚本路径：
[labctl.sh](labctl.sh)

## 1. 首次使用

```bash
cd /Users/pwl/Develop/SAO/sao
chmod +x labctl.sh
```

建议先激活 Python 环境（conda 或 venv），再执行依赖检查：

```bash
./labctl.sh check
```

## 2. 一键启动

```bash
cd /Users/pwl/Develop/SAO/sao
./labctl.sh start
```

启动成功后可访问：

1. Qwen API: http://127.0.0.1:8008/docs
2. Gradio UI: http://127.0.0.1:7860

## 3. 常用命令

```bash
./labctl.sh status
./labctl.sh logs
./labctl.sh stop
./labctl.sh restart
```

## 4. 环境变量

1. PYTHON_BIN: Python 可执行路径（默认 `python`）
2. QWEN_HOST / QWEN_PORT: Qwen 监听地址和端口（默认 `0.0.0.0:8008`）
3. GRADIO_HOST / GRADIO_PORT: Gradio 监听地址和端口（默认 `0.0.0.0:7860`）
4. QWEN_SERVICE_URL: Gradio 调用 Qwen 的 URL（默认 `http://127.0.0.1:8008/refine_prompt`）
5. QWEN_START_TIMEOUT: Qwen 启动等待秒数（默认 `600`）
6. GRADIO_START_TIMEOUT: Gradio 启动等待秒数（默认 `60`）

示例：

```bash
QWEN_PORT=8010 GRADIO_PORT=7861 ./labctl.sh start
QWEN_START_TIMEOUT=1800 ./labctl.sh start
PYTHON_BIN=/path/to/python ./labctl.sh start
```

## 5. 日志与运行文件

1. PID 文件：`.run/qwen.pid`、`.run/gradio.pid`
2. 日志文件：`experiments/logs/qwen.log`、`experiments/logs/gradio.log`

## 6. Qwen 启动失败 Debug

1. 查看状态：

```bash
./labctl.sh status
```

2. 查看日志：

```bash
./labctl.sh logs
# 或仅看 qwen

tail -f experiments/logs/qwen.log
```

3. 如果日志中出现 `Fetching ... files`，表示首次下载模型，属于正常现象。请提高超时后重试：

```bash
./labctl.sh stop
QWEN_START_TIMEOUT=1800 ./labctl.sh start
```

4. 若出现 `ModuleNotFoundError` / `ImportError`：确认环境后重新检查依赖：

```bash
conda activate sao
./labctl.sh check
```

5. 若端口被占用：

```bash
lsof -iTCP:8008 -sTCP:LISTEN -n -P
```

6. 若显存相关报错（CUDA/OOM）：关闭其他大模型进程，必要时改小负载或切 CPU。

## 7. 实际启动入口

1. Qwen 服务：[Qwen2AudioInstruct/qwen2audio_server.py](Qwen2AudioInstruct/qwen2audio_server.py)
2. Gradio 前端：[StableAudioOpen/gradio_lab.py](StableAudioOpen/gradio_lab.py)
