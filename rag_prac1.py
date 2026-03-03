from langchain_cohere import CohereEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Step 1: Document Loading
loader = PyPDFLoader("./EMPLOYEE_AGREEMENT.pdf")
pages = loader.load()
print(len(pages))

full_text =""
for page in pages:
  full_text += page.page_content

print("Pages", len(pages))
print("Lines" , len(full_text.split("\n")))
print("Words" , len(full_text.split(" ")))
print("Charecters", len(full_text))

# Step 2 Split the data into Chunks
#chunk_size defines the maximum number of characters in each text chunk (e.g., 300 characters per chunk).
#chunk_overlap repeats some characters (e.g., 30) between chunks to preserve context and avoid cutting important information at boundaries.
text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=30) # first 300 -> chuck 1 - 300, # second chunk 2 270 - 570
chunks = text_splitter.split_documents(pages)
print(len(chunks))
print(chunks[0])

# Step-3: Creating embeddings
embeddings = CohereEmbeddings(
    model="embed-english-v3.0",   # required
)
sample_embedding = embeddings.embed_query("You must follow the rules")
print(sample_embedding)

# Step-4: Storing in Vector Stores
emp_rules_db= Chroma.from_documents(chunks,
                                    embeddings,
                                    persist_directory="emp_rules_db"
                          )
emp_rules_db.persist()

# Step-5: Retrieval
retriever = emp_rules_db.as_retriever(search_kwargs={"k": 3})
result = retriever.invoke("What is the policy for sick leaves?")
for doc in result:
    print(doc.page_content)
    print("-----")