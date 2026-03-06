import subprocess
import os

from langchain_classic.chains.retrieval_qa.base import RetrievalQA
from langchain_community.chat_models import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

repo_url = "https://github.com/scikit-learn/scikit-learn.git"
folder = "scikit-learn"

subprocess.run(["git", "clone", "--filter=blob:none", "--no-checkout", repo_url])
os.chdir(folder)

subprocess.run(["git", "sparse-checkout", "init", "--cone"])
subprocess.run(["git", "sparse-checkout", "set", "examples/tree"])
subprocess.run(["git", "checkout"])

print("Downloaded only examples/tree folder.")

# Load vector DB
vectorstore = FAISS.load_local(
    "tree_code_index",
    OpenAIEmbeddings(),
    allow_dangerous_deserialization=True
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)

qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever
)

while True:
    query = input("\nAsk about the code: ")

    if query.lower() == "exit":
        break

    result = qa.run(query)
    print("\nAnswer:\n", result)
