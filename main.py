"""
@Author: bo.yang-a2302@aqara.com
@Date: 

"""
from langchain_community.vectorstores import FAISS
from langchain_core.messages import BaseMessage
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

import os
import json

from openai import api_key
from pydantic import SecretStr

# 1. 设置模型参数
EMBED_MODEL = "text-embedding-ada-002"
# EMBED_MODEL = "bge-m3:latest"
EMBED_BASE_URL = "http://43.173.190.109:1111/v1"
EMBED_API_KEY = "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"

LLM_MODEL = "Qwen/Qwen3-30B-A3B"
# LLM_MODEL = "qwen3:4b"
LLM_BASE_URL = "http://10.11.40.253:8002/v1"
LLM_API_KEY = "<KEY>"

VECTOR_DB_PATH = "./faiss_index"

# 2. 文档加载与分片（只执行一次）
def load_and_index_docs(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=100,
        chunk_overlap=10,
    )

    docs = [Document(page_content=chunk) for chunk in splitter.split_text(text)]
    # 向量化
    embedding = OpenAIEmbeddings(
        model=EMBED_MODEL,
        api_key=SecretStr(EMBED_API_KEY),
        base_url=EMBED_BASE_URL
    )
    vectorstore = FAISS.from_documents(docs, embedding)

    # 保存向量数据库
    vectorstore.save_local(VECTOR_DB_PATH)
    print("✅ 向量数据库已保存")

# 3. 加载向量库 & 检索
def retrieve_docs(querys, top_k=3):
    embedding = OpenAIEmbeddings(
        model=EMBED_MODEL,
        api_key=SecretStr(EMBED_API_KEY),
        base_url=EMBED_BASE_URL
    )
    vectorstore = FAISS.load_local(VECTOR_DB_PATH, embedding, allow_dangerous_deserialization=True)
    final_results = []
    ids = []
    for query in querys:
        results = vectorstore.similarity_search(query, k=top_k)
        for result in results:
            if result.id not in ids:
                final_results.append(result)
                ids.append(result.id)
    return final_results

def generate_mult_query(query):
    prompt_llm_query = f"""你是一个智能的查询扩展助手，你的任务是根据用户的原始问题，生成多个语义相近但表达不同的问句，以便更好地覆盖搜索意图和语义匹配。

    用户问题："{query}"

    请基于这个问题生成 5 个不同的、语义相近的自然语言问题，要求：
    1. 保持原意不变；
    2. 句式多样；
    3. 包含用户可能使用的不同提问方式或关键词。

    请以 JSON 数组格式返回，例如：
    [
      "相似问题1",
      "相似问题2",
      ...
    ]
    """
    # print("*" * 80)
    # print(prompt_llm_query)
    # print("*" * 80)

    llm = ChatOpenAI(
        model=LLM_MODEL,
        base_url=LLM_BASE_URL,
        api_key=SecretStr(LLM_API_KEY),
        temperature=0.7,
        top_p=0.8,
        max_tokens=1000,
        presence_penalty=2,
        # extra_body={"chat_template_kwargs": {"enable_thinking": False}}
    )
    result: BaseMessage = llm.invoke(prompt_llm_query)
    query = result.content
    query = json.loads(query)
    return query



# 4. 构造 Prompt 并用 qwen3 回答
def generate_answer(query, docs):
    context = "\n\n".join([doc.page_content for doc in docs])
    prompt = f"""你是一个智能助手，请根据以下内容回答用户问题：
========
{context}
========
问题：{query}
请用简洁和准确的语言回答："""
    llm = ChatOpenAI(
        model=LLM_MODEL,
        base_url=LLM_BASE_URL,
        api_key=SecretStr(LLM_API_KEY),
        temperature=0.0,
        top_p=0.8,
        max_tokens=32768,
        presence_penalty=1.5,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}}
    )
    return llm.invoke(prompt).content

# ======== 主流程（第一次运行时执行分片）========
if __name__ == "__main__":
    file_path = "C:/Users/admin/Desktop/test2.txt"  # 替换为你的文档路径
    if not os.path.exists(VECTOR_DB_PATH):
        load_and_index_docs(file_path)

    user_query = "这产品都适合哪些人"
    user_querys = generate_mult_query(user_query)
    relevant_docs = retrieve_docs(user_querys)
    answer = generate_answer(user_query, relevant_docs)

    print("\n💡 回答：", answer)