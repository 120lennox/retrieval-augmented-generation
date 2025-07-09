import os
from langchain_openai import ChatOpenAI

os.environ["GITHUB_TOKEN"] = "your access token"  # access token removed

llm = ChatOpenAI(
    model="openai/gpt-4.1",  # or whatever model name works
    openai_api_base="https://models.github.ai/inference",
    openai_api_key=os.environ["GITHUB_TOKEN"],
    temperature=0.7
)

prompt = input("prompt: ")
response = llm.invoke(prompt)
print(response.content)