import os
os.environ["CUDA_VISIBLE_DEVICES"] = "5,6"
 
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import pipeline
from langchain import HuggingFacePipeline
from langchain_core.prompts import ChatPromptTemplate
 
 
tokenizer = AutoTokenizer.from_pretrained("/home/cloud/yuwendi/qwen/Qwen-7B-Chat",
                                              trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("/home/cloud/yuwendi/qwen/Qwen-7B-Chat",
                                                 device_map="auto",
                                                 trust_remote_code=True).eval()
 
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    # max_length=4096,
    # max_tokens=4096,
    max_new_tokens=512,
    top_p=1,
    repetition_penalty=1.15
)
llama_model = HuggingFacePipeline(pipeline=pipe)
 
prompt = ChatPromptTemplate.from_template("请编写一篇关于{topic}的中文小故事，不超过100字")
chain = prompt | llama_model
res = chain.invoke({"topic": "小白兔"})
print(res)