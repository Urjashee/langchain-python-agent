from dotenv import load_dotenv
from langchain import hub
from langchain.agents import AgentExecutor
from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_csv_agent
from langchain_experimental.tools import PythonREPLTool
from langchain.agents import create_openai_functions_agent
from langchain_openai import ChatOpenAI, OpenAI

load_dotenv()
tools = [PythonREPLTool()]


def main():
    print("Hello World")
    llm = ChatOpenAI(model='gpt-3.5-turbo', temperature=0)
    instructions = """You are an agent designed to write and execute python code to answer questions.
    You have access to a python REPL, which you can use to execute python code.
    If you get an error, debug your code and try again.
    Only use the output of your code to answer the question.
    You might know the answer without running any code, but you should still run the code to get the answer.
    If it does not seem like you can write code to answer the question, just return "I don't know" as the answer.
    """
    base_prompt = hub.pull("langchain-ai/openai-functions-template")
    prompt = base_prompt.partial(instructions=instructions)

    agent = create_openai_functions_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
    input_prompt = "What is the 10th fibonacci number?"

    # agent_executor.invoke({"input": input_prompt})

    csv_agent = create_csv_agent(
        llm,
        "episode-info.csv",
        verbose=True,
        agent_type=AgentType.OPENAI_FUNCTIONS,
        allow_dangerous_code=True
    )

    # query = "which writer wrote the most episodes in episode-info.csv? how many episodes did he write"
    query = "print season ascending order of the number of episodes"
    csv_agent.run(query)


if __name__ == '__main__':
    main()
