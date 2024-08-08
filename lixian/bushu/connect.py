from openai import OpenAI

# Modify OpenAI's API key and API base to use vLLM's API server.
openai_api_key = "EMPTY"
openai_api_base = "http://0.0.0.0:8085/v1/"
client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)
completion = client.completions.create(model="qwen",
                                        prompt="你是一个小说家，你现在要写一篇小说。从前，有一个老人住在山里",max_tokens=300,temperature=0.6)
print("Completion result:", completion.choices[0].text)
