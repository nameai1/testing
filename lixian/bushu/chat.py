from modelscope import AutoModelForCausalLM, AutoTokenizer

model_path = "/home/cloud/yuwendi/qwen/Qwen-7B-Chat"

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", trust_remote_code=True).eval()

history=None

while True:
    message = input('User:')
    response, history = model.chat(tokenizer, message, history=history)
    print('System:',end='')
    print(response)

