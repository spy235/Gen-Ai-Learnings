# Download the PDF
# Updated imports for LangChain
from langchain_classic.chains.retrieval_qa.base import RetrievalQA
from langchain_community.chat_models import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Document Loading
loader = PyPDFLoader("COI.pdf")
pages = loader.load()

full_text = ""
for page in pages:
    full_text += page.page_content

print("Pages:", len(pages))
print("Lines:", len(full_text.split("\n")))
print("Words:", len(full_text.split(" ")))
print("Characters:", len(full_text))

# Split the data into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=400)
chunks = text_splitter.split_documents(pages)
print("Number of chunks:", len(chunks))

# Embeddings and Vector DB
embeddings = OpenAIEmbeddings()

coi_db = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="coi_db"
)
coi_db.persist()

# Retrieval Q&A Chain
llm = ChatOpenAI(temperature=0)  # Use ChatOpenAI instead of OpenAI

Q_AChain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",  # Pass all chunks into the prompt
    retriever=coi_db.as_retriever()
)