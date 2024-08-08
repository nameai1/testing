from langchain_community.vectorstores import Chroma
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.llms import Tongyi
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
model_name = r"/home/cloud/yuwendi/embedding/maidalun/bce-embedding-base_v1"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': False}
embeddings = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

db = Chroma(persist_directory="/home/cloud/yuwendi/lixian/text.txt", embedding_function=embeddings)

retriever = db.as_retriever()
qa = RetrievalQA.from_chain_type(llm=llama_model, retriever=retriever)

query = "浩浩的科研笔记的原力等级是多少？"
print(qa.run(query))
