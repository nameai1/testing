from transformers import AutoModelForCausalLM, AutoTokenizer
from yuwendi.lixian.function.function_call import CourseDatabase, CourseOperations
from vllm import LLM, SamplingParams
import yuwendi.lixian.rag.RAG as RAG


model_path = "/home/cloud/yuwendi/qwen/Qwen-7B-Chat"

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", trust_remote_code=True).eval()


react_stop_words = [
    tokenizer.encode('Observation:'),
    tokenizer.encode('Observation:\n'),
]
history=None
import json
def generate_action_prompt(query):
    """
    根据用户查询生成最终的动作提示字符串。
    函数内部直接引用全局变量 TOOLS, TOOL_DESC, 和 PROMPT_REACT.
    参数：
    - query: 用户的查询字符串。
    返回：
    - action_prompt: 格式化后的动作提示字符串。
    """
 
    tool_descs = []
    tool_names = []
 
    for info in TOOLS:
        tool_descs.append(
            TOOL_DESC.format(
                name_for_model = info['name_for_model'],
                name_for_human = info['name_for_human'],
                description_for_model = info['description_for_model'],
                parameters = json.dumps(info['parameters'], ensure_ascii=False),
            )
        )
        tool_names.append(info['name_for_model'])
 
    tool_descs_str = '\n\n'.join(tool_descs)
    tool_names_str = ','.join(tool_names)
 
    action_prompt = PROMPT_REACT.format(tool_descs=tool_descs_str, tool_names=tool_names_str, query=query)
    return action_prompt

import json

def get_prompt(query):
    PROMPT_REACT = """
    Use the following format:
    Question: the input question you must answer
    Thought: you should always think about what to do
    Begin!
    Question:{query}"""

    reason_prompt = PROMPT_REACT.format(query=query)
    return reason_prompt

def generate_action_prompt(query):
    """
    根据用户查询生成最终的动作提示字符串。
    函数内部直接引用全局变量 TOOLS, TOOL_DESC, 和 PROMPT_REACT.
    参数：
    - query: 用户的查询字符串。
    返回：
    - action_prompt: 格式化后的动作提示字符串。
    """
 
    tool_descs = []
    tool_names = []
 
    for info in TOOLS:
        tool_descs.append(
            TOOL_DESC.format(
                name_for_model = info['name_for_model'],
                name_for_human = info['name_for_human'],
                description_for_model = info['description_for_model'],
                parameters = json.dumps(info['parameters'], ensure_ascii=False),
            )
        )
        tool_names.append(info['name_for_model'])
 
    tool_descs_str = '\n\n'.join(tool_descs)
    tool_names_str = ','.join(tool_names)
 
    action_prompt = PROMPT_REACT.format(tool_descs=tool_descs_str, tool_names=tool_names_str, query=query)
    return action_prompt

TOOLS = [
    {
        'name_for_human': '课程信息数据库',
        'name_for_model': 'CourseDatabase',
        'description_for_model': '课程信息数据库存储有各课程的详细信息,包括目前的上线课时，每周更新次数以及每次更新的小时数。通过输入课程名称，可以返回该课程的详细信息。',
        'parameters': [{
            'name': 'course_query',
            'description': '课程名称,所需查询信息的课程名称',
            'required': True,
            'schema': {
                'type': 'string'
            },
        }],
    },
    # 其他工具的定义可以在这里继续添加
]

# 将一个插件的关键信息拼接成一段文本的模板
TOOL_DESC = """{name_for_model}: Call this tool to interact with the {name_for_human} API. What is the {name_for_human} API useful for? {description_for_model} Parameters:{parameters}
"""
 
PROMPT_REACT = """Answer the following questions as best you con. You have access to the following
{tool_descs}
Use the following format:
Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can be repeated zero or more times)
Begin!
Question: {query}"""

react_stop_words = [
    tokenizer.encode('Observation:'),
    tokenizer.encode('Observation:\n'),
]

query = "先帮我查询一下大模型技术实战这个课程目前更新了多少节，今晚我直播了一节新课，请你帮我更新一下"
RAG = generate_action_prompt(query)
print(RAG)

# 使用action_prompt生成回复
response, history = model.chat(tokenizer, RAG, history=None, \
                              stop_words_ids=react_stop_words)
print(response)

def parse_plugin_action(text: str):
    """
    解析模型的ReAct输出文本提取名称及其参数。
    参数：
    - text： 模型ReAct提示的输出文本
    返回值：
    - action_name: 要调用的动作（方法）名称。
    - action_arguments: 动作（方法）的参数。
    """
    # 查找“Action:”和“Action Input：”的最后出现位置
    action_index = text.rfind('\nAction:')
    action_input_index = text.rfind('\nAction Input:')
    observation_index = text.rfind('\nObservation:')
 
    # 如果文本中有“Action:”和“Action Input：”
    if 0 <= action_index < action_input_index:
        if observation_index < action_input_index:
            text = text.rstrip() + '\nObservation:'
            observation_index = text.rfind('\nObservation:')
 
    # 确保文本中同时存在“Action:”和“Action Input：”
    if 0 <= action_index < action_input_index < observation_index:
        # 提取“Action:”和“Action Input：”之间的文本为动作名称
        action_name = text[action_index + len('\nAction:'):action_input_index].strip()
        # 提取“Action Input：”之后的文本为动作参数
        action_arguments = text[action_input_index + len('\nAction Input:'):observation_index].strip()
        return action_name, action_arguments
 
    # 如果没有找到符合条件的文本，返回空字符串
    return '', ''

import json
def execute_plugin_from_react_output(response):
    """
    根据模型的ReAct输出执行相应的插件调用，并返回调用结果。
    参数：
    - response: 模型的ReAct输出字符串。
    返回：
    - result_dict: 包括状态码和插件调用结果的字典。
    """
    # 从模型的ReAct输出中提取函数名称及函数入参
    plugin_configuration = parse_plugin_action(response)
    first_config_line = plugin_configuration[1:][0].split('\n')[0]
    config_parameters = json.loads(first_config_line)
    result_dict = {"status_code": 200}
 
    for k, v in config_parameters.items():
        if k in TOOLS[0]["parameters"][0]['name']:
            # 通过eval函数执行存储在字符串中的python表达式，并返回表达式计算结果。其执行过程实质上是实例化类
            tool_instance = eval(TOOLS[0]["name_for_model"])()
            # 然后通过getattr函数传递对象和字符串形式的属性或方法名来动态的访问该属性和方法h
            tool_func = getattr(tool_instance, k)
            # 这一步实际上执行的过程就是：course_db,course_query('大模型技术实战')
            tool_result = tool_func(v)
            result_dict["result"] = tool_result
            return result_dict
 
    result_dict["status_code"] = 404
    result_dict["result"] = "未找到匹配的插件配置"
    return result_dict
 
 
tool_result = execute_plugin_from_react_output(response)
print(tool_result)

response += " " + str(tool_result)
print(response)
response, history = model.chat(tokenizer, response, history=history, \
                              stop_words_ids=react_stop_words)
print(response)