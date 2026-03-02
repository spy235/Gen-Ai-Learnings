from langchain_cohere import ChatCohere
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

from langchain_core.prompts import PromptTemplate

llm=ChatCohere(temperature=0.1)
llm=ChatOpenAI(
    model="gpt-4o-mini",   # cheapest OpenAI chat model
    temperature=0.1)
template = "List down the historically significant steps in the field of {filed_name}"
prompt = PromptTemplate(
    input_variables=["filed_name"],
    template=template,
)

chain= llm | prompt
#chian= prompt | llm
result=chain.invoke("Machine Learning")
print(result)

chains
from langchain_core.output_parsers import StrOutputParser
#llm=OpenAI(temperature=0.5)
llm=ChatCohere(temperature=0.5)

# chain 1
book_name_prompt_template = PromptTemplate(
    input_variables=["theme"],
    template="""Please provide a simple list of ten well-known books that center around the theme of {theme}.
Do not include book descriptions."""
)

# Build chain
book_name_chain = book_name_prompt_template | llm | StrOutputParser()

# Get list of books
books_text = book_name_chain.invoke({"theme": "personality development"})
print("Books:\n", books_text)

# chain 2
book_summary_prompt_template = PromptTemplate(
    input_variables=["book_names_list"],
    template="""Please take any one book from the list {book_names_list}.
Mention the book title.
Provide a comprehensive summary in three sections,
with three summary points per section."""
)

# Build chain
book_summary_chain = book_summary_prompt_template | llm | StrOutputParser()

# Summarize the books
book_summary = book_summary_chain.invoke({"book_names_list": "personal finance"})

print("\nBook Summary:\n", book_summary)




# squential chain ................................................

llm = ChatCohere( temperature=0.1)

# Step 1: Book names
book_name_prompt_template = PromptTemplate(
    input_variables=["theme"],
    template="Provide 10 book titles about {theme}, no descriptions."
)
book_name_chain = book_name_prompt_template | llm | StrOutputParser()

# Step 2: Book summary
book_summary_prompt_template = PromptTemplate(
    input_variables=["book_names_list"],
    template="Pick one book from {book_names_list} and summarize it in 3 sections with 3 points each."
)
book_summary_chain = book_summary_prompt_template | llm | StrOutputParser()

# Sequential Chain
# Chain them together
book_chain = book_name_chain | book_summary_chain

# Run
book_result = book_chain.invoke({"theme": "Personal Finance"})
print(book_result)