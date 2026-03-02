import random as rand
import requests
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader, WikipediaLoader, PyPDFLoader, BSHTMLLoader,YoutubeLoader
from langchain_cohere import ChatCohere
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field


email_location="https://raw.githubusercontent.com/venkatareddykonasani/Datasets/master/Customer_Emails/Mail"+str(rand.randint(1,5))+".txt"
print(email_location)

loader = WebBaseLoader(email_location)
loaded_text= loader.load()
print(type(loaded_text))
final_mail=loaded_text[0].page_content
print(final_mail)


class EmailResponse(BaseModel):
  Email_Language: str= Field(description= "The Original Language of the Email")
  Customer_ID: str= Field(description= "The Customer ID mentioned in the mail")
  English_email: str= Field(description= "The email after translating to English")
  Summary: str= Field(description= "A 4 bullets point summary of the email")
  Reply: str= Field(description= "A polite 2 line reply to the email")

custom_output_parser= PydanticOutputParser(pydantic_object=EmailResponse)
print(custom_output_parser.get_format_instructions())


template="""
Take the email as input. Email text is {email}
{format_instructions}
"""
prompt=PromptTemplate(template=template,
                      input_variables=["email","format_instructions"])

#llm=OpenAI(temperature=0)
llm=ChatCohere()

chain=prompt | llm

result=chain.invoke({"email":final_mail,
                     "format_instructions":custom_output_parser.get_format_instructions()})
print(result)