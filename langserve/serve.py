
import os
import dotenv
import uvicorn

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langserve import add_routes
from fastapi import FastAPI

dotenv.load_dotenv()

os.environ['OPENAI_API_KEY'] = os.getenv("LANGCHAIN_API_KEY")

model_name = "gpt-4o"

model = ChatOpenAI(
    model=model_name
)

template = """你是一个专业的中英翻译，请根据用户的输出将其进行准确的翻译，翻译请遵循以下几个规则：
1. 如果用户输入的是一个中文词，请翻译为英文并输出其音标
2. 如果用户输入的是一段中文文本，请直接翻译为英文
3. 如果用户输入的是英文单词，请输出其音标以及中文翻译
4. 如果用户输入的是一段英文文本，请输出其中文翻译
5. 遵循以上输出规则，切勿生成额外的东西"""

messages = [
    ("system", template),
    ("user", "{text}")
]
prompt = ChatPromptTemplate.from_messages(messages)
parser = StrOutputParser()
chain = prompt | model | parser

app = FastAPI(
    title="Langchain Server",
    version="1.0",
    description="A simple Langchain Server"
)

add_routes(
    app,
    chain,
    path="/chain"
)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
