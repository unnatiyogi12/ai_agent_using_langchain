import os
import certifi
import requests
from dotenv import load_dotenv

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_classic import hub
from langchain_classic.agents import create_react_agent, AgentExecutor
from langchain_ollama import ChatOllama
from langchain.tools import tool   #for custom tools we can use tools

llm = ChatOllama(
    model="qwen2.5:3b",
    temperature=0
)

response = llm.invoke("What is an AI agent?")
print(response.content)
os.environ["SSL_CERY_FILE"] = certifi.where()
load_dotenv()

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
search_tool = TavilySearchResults(max_results=2)
result = search_tool.invoke("what is the lastest news on AI")
result
from langchain_classic import hub

print("Hub imported successfully")
# prompt
from langsmith import Client

client = Client()

prompt = client.pull_prompt(
    "hwchase17/react",
    dangerously_pull_public_prompt=True
)
# print(type(prompt))

prompt
# tools

# custom tool
@tool
def get_weather_data(city:str) -> str:
    'fetch current weather data'
# use weather data api key
    return(
        print("city data printed")
    )

tools = [search_tool, get_weather_data]
# create agent

agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=prompt
)

# agent executor

'''
1. sends input and previous messages to the agent
2. gets the next action from agent
3. Executes that tool with provided input
4. Adds thr tools observation back to the history
5. loops again with updated history until the agents says final answer
 
'''
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True   #it show all the logs if you marked it true otherwise dont show the logs just provide the output
)
# run

response = agent_executor.invoke({
    "input":(
        'Find the capital of India'
    )
})
print(response)
