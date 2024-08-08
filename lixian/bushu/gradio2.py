import os

import gradio as gr
from openai import OpenAI

openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8080/v1/"

SYSTEM_PROMPT = """
你是一名资深教师，你叫“同学小张”，用户会给你一个提示，你根据用户给的提示，来为用户设计关于此课程的学习大纲。
你必须遵循以下原则：
1. 你有足够的时间思考，确保在得出答案之前，你已经足够理解用户需求中的所有关键概念，并给出关键概念的解释。
2. 输出格式请使用Markdown格式, 并保证输出内容清晰易懂。
3. 至少输出10章的内容, 每章至少有5个小节

不要回答任何与课程内容无关的问题。
"""
history =[]
def predict(message,history1):
    client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)

    response =  client.chat.completions.create(
        model="qwen",
        messages=history+[ {
                    'role': 'teacher',
                    'content': SYSTEM_PROMPT
                },
                {
                    'role': 'user',
                    'content': message
                }],
        max_tokens=300,
        temperature=0.6,
        stream=True
    )
    text = ""
    for msg in response:
        delta = msg.choices[0].delta
        if delta.content:
            text_delta = delta.content
            text = text + text_delta
            yield text

if __name__ == "__main__":
    demo = gr.ChatInterface(predict).queue()
    demo.launch()