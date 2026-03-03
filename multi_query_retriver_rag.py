# Import necessary LangChain components
from langchain_classic.chains.retrieval_qa.base import RetrievalQA
from langchain_classic.retrievers import MultiQueryRetriever, ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import LLMChainExtractor
from langchain_cohere import CohereEmbeddings, ChatCohere
from langchain_community.document_loaders import WikipediaLoader
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
import logging


# ==============================
# 1️⃣ Document Loading
# ==============================

# Load documents from Wikipedia related to "MS Dhoni"
# WikipediaLoader fetches page content as LangChain documents
loader = WikipediaLoader(query="MS Dhoni")
documents = loader.load()


# ==============================
# 2️⃣ Text Splitting
# ==============================

# Split large documents into smaller chunks
# chunk_size = 500 characters
# chunk_overlap = 50 characters (helps maintain context between chunks)
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

docs = text_splitter.split_documents(documents)

# Print total number of chunks created
print(len(docs))


# ==============================
# 3️⃣ Embeddings + Vector Database
# ==============================

# Create embedding model using Cohere
# Converts text into numerical vectors
embeddings = CohereEmbeddings(
    model="embed-english-v3.0",   # Required model name
)

# Store document chunks in Chroma vector database
# persist_directory saves DB locally so it can be reused later
embeddings_db = Chroma.from_documents(
    docs,
    embeddings,
    persist_directory="wiki_db"
)

# Persist (save) vector DB to disk
embeddings_db.persist()


# ==============================
# 4️⃣ LLM Setup
# ==============================

# Initialize Chat model (temperature=0 → deterministic output)
llm = ChatCohere(temperature=0)


# ==============================
# 5️⃣ MultiQueryRetriever
# ==============================

# MultiQueryRetriever:
# Uses LLM to generate multiple variations of a question
# This improves retrieval quality by searching using multiple query forms
llm_based_retriver = MultiQueryRetriever.from_llm(
    retriever=embeddings_db.as_retriever(),
    llm=llm
)

print(llm_based_retriver)

# Enable logging to see how multiple queries are generated internally
logging.basicConfig(level=logging.INFO)
logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)


# Example questions
question1 = "What is the DOB of Dhoni?"
question2 = "What Sport does Dhoni Play?"


# ==============================
# (Optional) Testing Retriever Alone
# ==============================

# These lines retrieve relevant documents without QA chain
# rel_docs1 = llm_based_retriver.invoke(question1)
# rel_docs2 = llm_based_retriver.invoke(question2)
#
# print(rel_docs1)
# print(rel_docs2)


# ==============================
# 6️⃣ Contextual Compression (Optional Optimization)
# ==============================

# Compression reduces retrieved document size
# Only keeps relevant parts of documents using LLM

# llm = ChatCohere(temperature=0)
#
# compressor = LLMChainExtractor.from_llm(llm)
#
# compression_retriever = ContextualCompressionRetriever(
#     base_compressor=compressor,
#     base_retriever=embeddings_db.as_retriever()
# )
#
# compressed_docs = compression_retriever.invoke(question1)
# print(compressed_docs[0].metadata)


# ==============================
# 7️⃣ Retrieval QA Chain
# ==============================

# RetrievalQA:
# 1. Retrieves relevant documents
# 2. Sends them to LLM
# 3. Generates final answer

llm = ChatCohere(temperature=0)

Q_AChain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    # "stuff" → inserts ALL retrieved documents into the prompt
    # Other types: map_reduce, refine, etc.
    retriever=llm_based_retriver
)

query = "What is the DOB of Dhoni?"

# Execute QA chain
# It returns a dictionary with 'result' and optionally source docs
docs = Q_AChain({"query": query})

# Print final answer
print(docs["result"])


# ==============================
# 8️⃣ Inspect Prompt Used Internally
# ==============================

# Print the prompt template used by the chain
print(Q_AChain.combine_documents_chain.llm_chain.prompt)

# Print message structure of the prompt (system + human messages)
print(Q_AChain.combine_documents_chain.llm_chain.prompt.messages)