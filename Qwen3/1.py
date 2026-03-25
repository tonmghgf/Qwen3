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