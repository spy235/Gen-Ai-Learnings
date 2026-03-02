from langchain_classic.chains.llm import LLMChain
from langchain_cohere import ChatCohere
from langchain_community.document_loaders import WikipediaLoader
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field

loader=WikipediaLoader(query="TCS",  load_max_docs=1)
company_documents=loader.load()
print(company_documents)

class CompanyProfile(BaseModel):
  Company_Name: str= Field(description= "The Name of the Company")
  Founder: str= Field(description= "The Founder of the Company")
  Start_Date: str= Field(description= "The date or the founding year of the compnay")
  Revenue: int= Field(description= "The Revenue of the company")
  Employees: str= Field(description= "How many employees are working in it")
  Summary: str= Field(description= "Provide a brief 4-line summary of the company")

custom_output_parser= PydanticOutputParser(pydantic_object=CompanyProfile)
print(custom_output_parser.get_format_instructions())

template="""
Take the company wiki page information as input
Company Details from Wikipedia:{wiki_page_info}
{format_instructions}
"""
prompt=PromptTemplate(template=template,
                      input_variables=["wiki_page_info","format_instructions"])

llm=ChatCohere()

chain=LLMChain(prompt=prompt,
               llm=llm)

result=chain.invoke({"wiki_page_info":company_documents,
              "format_instructions":custom_output_parser.get_format_instructions()})
print(result["text"])