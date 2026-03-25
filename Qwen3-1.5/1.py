# 1. 加载需要的工具包
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# 2. 自动选择设备（GPU 有就用，没有就用 CPU）
device = "cuda" if torch.cuda.is_available() else "cpu"

# 3. 模型路径（你下载好的 Qwen2.5-0.5B-Instruct）
model_path = "qwen/Qwen1.5-0.5B-Instruct"

# 4. 加载模型
print("正在加载模型...")
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_path)


# 5. 对话函数
def chat(message):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": message}
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    model_inputs = tokenizer([text], return_tensors="pt").to(device)

    # 生成回答
    generated_ids = model.generate(
        model_inputs.input_ids,
        max_new_tokens=512
    )

    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response


# 6. 测试对话
if __name__ == "__main__":
    print("模型加载完成！输入内容对话，输入 exit 退出。\n")
    while True:
        user_input = input("我：")
        if user_input.lower() == "exit":
            print("对话结束")
            break
        resp = chat(user_input)
        print("AI：", resp, "\n")