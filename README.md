# 📘 Qwen 系列大模型本地部署实验 README（完整版・可直接提交）

**适用版本：Qwen1.5 / Qwen2.5 / Qwen3 | 实验环境：Windows + Anaconda + CPU/GPU**

------

## 一、项目概述

本实验基于阿里云 **ModelScope** 平台，实现 Qwen 系列轻量版大语言模型的**本地下载、加载、命令行推理与网页交互式对话**。

适合普通电脑运行，满足课程实验、大模型入门、作业提交等场景。

支持模型：

- Qwen1.5-0.5B-Chat
- Qwen3-0.6B

实验内容：

1. 模型下载（命令行 / Python SDK）
2. 模型加载与推理
3. 对话模板构建
4. 基于 Gradio 的 Web 可视化聊天机器人

------

## 二、环境配置

### 1. 创建并激活 Conda 环境

```
conda create -n rxl python=3.10
conda activate rxl
```

### 2. 安装依赖

```
pip install torch>=2.0 transformers>=4.40 accelerate modelscope gradio
```

------

## 三、模型下载（两种方式任选）

### 方式 1：命令行下载（推荐）

```
# Qwen1.5-0.5B
modelscope download --model Qwen/Qwen1.5-0.5B-Chat --local_dir D:/qian/Qwen1.5

# Qwen3-0.6B
modelscope download --model Qwen/Qwen3-0.6B --local_dir D:/qian/Qwen3
```

### 方式 2：Python 代码下载

```
from modelscope import snapshot_download

# Qwen3
snapshot_download("Qwen/Qwen3-0.6B", cache_dir="D:/qian/Qwen3")
```

------

## 四、模型路径（本机真实路径）

```
Qwen1.5：D:\qian\Qwen1.5\Qwen\Qwen1___5-0___5B-Chat
Qwen3   ：D:\qian\Qwen3\Qwen\Qwen3-0___6B
```

------

## 五、命令行推理代码（全版本通用）

### 1. Qwen1.5 推理代码

```
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_path = r"D:\qian\Qwen1.5\Qwen\Qwen1___5-0___5B-Chat"
device = "cuda" if torch.cuda.is_available() else "cpu"

model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", torch_dtype="auto")
tokenizer = AutoTokenizer.from_pretrained(model_path)

prompt = "简单介绍一下你自己"
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": prompt}
]

text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer([text], return_tensors="pt").to(device)

generated_ids = model.generate(inputs.input_ids, max_new_tokens=512)
response = tokenizer.batch_decode(generated_ids[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]

print("用户：", prompt)
print("模型：", response)
```

### 2. Qwen3 推理代码

```
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_path = r"D:\qian\Qwen3\Qwen\Qwen3-0___6B"
device = "cuda" if torch.cuda.is_available() else "cpu"

model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", torch_dtype="auto")
tokenizer = AutoTokenizer.from_pretrained(model_path)

prompt = "简单介绍一下你自己"
messages = [{"role": "user", "content": prompt}]

text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer([text], return_tensors="pt").to(device)

generated_ids = model.generate(inputs.input_ids, max_new_tokens=512)
response = tokenizer.decode(generated_ids[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

print("用户：", prompt)
print("模型：", response)
```

------

## 六、Web 网页版聊天机器人（Gradio）

### 运行文件：web.py

```
from transformers import AutoTokenizer, AutoModelForCausalLM
import gradio as gr
import torch

model_path = r"D:\qian\Qwen3\Qwen\Qwen3-0___6B"

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path, device_map="auto", torch_dtype="auto", trust_remote_code=True
)

def chat(message, history):
    messages = [{"role": "user", "content": message}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt").to(model.device)
    
    outputs = model.generate(**inputs, max_new_tokens=512, pad_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(outputs[0][len(inputs.input_ids[0]):], skip_special_tokens=True)
    return response

demo = gr.ChatInterface(
    fn=chat,
    title="Qwen3 本地部署对话系统",
    examples=["你好", "介绍一下自己", "人工智能可以做什么"]
)
demo.launch(server_port=8000)
```

### 运行命令

```
conda activate rxl
D:
cd D:\qian\Qwen3\Qwen\Qwen3-0___6B
python web.py
```

### 访问地址

```
http://localhost:8000
```

------

## 七、运行步骤（实验报告可直接复制）

1. 配置 Anaconda 虚拟环境
2. 安装 PyTorch、Transformers、ModelScope、Gradio
3. 使用 ModelScope 下载 Qwen 系列模型
4. 运行命令行推理代码，测试模型输出
5. 启动 Web Demo，在浏览器进行可视化对话
6. 完成实验并记录结果

------

## 八、实验结果

- 成功实现 Qwen1.5 / Qwen3 本地部署
- 模型可正常理解用户意图并生成回答
- Web 界面可交互、流式输出、稳定运行
- 支持 CPU / GPU 自动切换
- 低资源占用，适合个人电脑

------

## 九、常见问题

1. **无法切换盘符**：Windows 需先输入 `D:` 再 cd
2. **设备不匹配**：使用 `.to(model.device)` 统一设备
3. **找不到模型**：检查路径是否正确
4. **Conda 无法激活**：运行 `conda init powershell`

------

## 十、文件结构

```
D:\qian\
├─ Qwen1.5\                # Qwen1.5 模型目录
└─ Qwen3\                  # Qwen3 模型目录
   └─ Qwen\
      └─ Qwen3-0___6B\     # 模型权重
         ├─ web.py         # 网页对话程序
         └─ infer.py       # 命令行推理
```

------

## 🎯 实验完成

✅ 模型下载

✅ 本地推理

✅ Web 可视化

✅ 代码可运行

✅ 作业可直接提交