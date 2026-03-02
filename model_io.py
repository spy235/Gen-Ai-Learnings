from tempfile import template

from langchain_cohere import ChatCohere
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader, WikipediaLoader, PyPDFLoader, BSHTMLLoader,YoutubeLoader

from langchain_community.document_loaders import CSVLoader
from langchain_community.document_loaders import TextLoader


# read .md file -------------------------------------------------------------------------------------------------
loader = TextLoader("./Mobile.md")
loaded_text= loader.load()
print(type(loaded_text))
print(loaded_text)

llm=ChatCohere()

template="""
read the following review and extract the following information:
Name of the product
Brand of the product
Price of the product
Rating of the product

review is given here : {input_review}
"""
prompt=PromptTemplate(template=template,
                       input_variables=["input_review"])
chain= prompt | llm
result=chain.invoke({"input_review":loaded_text})
print(result)

# read .txt file ----------------------------------------------------------------------------------------

loader = TextLoader("./Leads.csv")
loaded_text= loader.load()
print(type(loaded_text))
print(loaded_text)

llm=ChatCohere()

template="""
read the following mail and extract the following information:
Name of the person
Main Point of the mail
Ticket-id
The mail is given here : {input_mail}
"""
prompt=PromptTemplate(template=template,
                       input_variables=["input_mail"])

chain=prompt | llm

result=chain.invoke({"input_mail":loaded_text})
print(result)

## WebBaseLoader ----------------------------------------------------------------------------------------------------------------------------------------------------------
import os
os.environ["USER_AGENT"] = "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
loader = WebBaseLoader("https://www.amazon.in/Rasayanam-Multivitamin-ingredients-Probiotic-Ashwagandha/dp/B0D7J183WS/ref=sr_1_13?adgrpid=1327112148528381&dib=eyJ2IjoiMSJ9.GUgtcHCpC9pi1lJnjhuNikZp0a5Um3RfyFYqf1pSKWavCptnN0vLXVvfh16aPq2ZZk_G5s_V_f87OYt6c14rqui_Njt39HZ3OEfewuk94T9gkRzGY-3EshI7X9VsXFNG2hpgF_Mey7SnSrucAiK2A9q7vrtouuXTi3vIhHTSJy-qF-_nENk9WcRn1l8BNK-loRbK-nosugcMj42z4vOZ-OgQ6CTZzqTdaZYKjNcspQM.Ybquk52SA4WuP-WRoWAHdq0KaMtWWuO0za3npQS5kQo&dib_tag=se&hvadid=82944775541359&hvbmt=bb&hvdev=c&hvlocphy=149884&hvnetw=o&hvqmt=b&hvtargid=kwd-82945393014646%3Aloc-90&hydadcr=5626_2377281&keywords=join%2Bamazon%2Bprime&mcid=3861a9d242543041b997efa1f39279d3&msclkid=a6c84c79ab7012ce311d25dde81fc7e0&qid=1772372850&sr=8-13&th=1")


url = "https://raw.githubusercontent.com/venkatareddykonasani/Datasets/master/Mobile_Phone_Review/Mobile.md"

loader = WebBaseLoader(url)
docs = loader.load()

# print(docs[0].page_content[:1000])

llm=ChatCohere()

template="""
read the following data summarize it into 4 bullet points
data is given here : {input_data}
"""

prompt=PromptTemplate(template=template,
                       input_variables=["input_data"])

chain=prompt | llm

result=chain.invoke({"input_data":docs})
print(result)


## Wiki Pedia Loader.......................................................................................
loader=WikipediaLoader(query="Sundar Pichai", load_max_docs=1)
wiki_file_data=loader.load()
print(type(wiki_file_data))
print(wiki_file_data)

llm=ChatCohere()

template="""
Who is the ceo of google in provided input: {input_data}
"""

prompt=PromptTemplate(template=template,
                       input_variables=["input_data"])

chain=prompt | llm

result=chain.invoke({"input_data":wiki_file_data})
print(result)

## Pdf reader..........................................................................
loader=PyPDFLoader(file_path="./Regulatory_Rules_in_Credit_Risk_Models.pdf")
pdf_file_data=loader.load()
print(type(pdf_file_data))
print(pdf_file_data)

llm=ChatCohere()

template="""
summarize this : {input_data}
"""

prompt=PromptTemplate(template=template,
                       input_variables=["input_data"])

chain=prompt | llm

result=chain.invoke({"input_data":pdf_file_data})
print(result)


## HTML ---------------------------------------------------------------------------------------------------
loader = BSHTMLLoader(
    "./South_Asia_Global_Debt_Summary.html",
    bs_kwargs={"features": "html.parser"}
)

html_file_data = loader.load()

print(type(html_file_data))
print(html_file_data)

llm = ChatCohere()

template = """
give xpath only for the element showing an increase of {percentage}% in {year}
from the following data:
{input_data}
"""

prompt = PromptTemplate(
    template=template,
    input_variables=["input_data", "percentage", "year"]
)

chain = prompt | llm

result = chain.invoke({
    "input_data": html_file_data,
    "percentage": 23,
    "year": 2006
})

print(result)

loader = YoutubeLoader.from_youtube_url(
    "https://www.youtube.com/watch?v=fbQvVS_8ZNI"
)

youtube_file_data = loader.load()
print(type(youtube_file_data))

llm = ChatCohere()

template = '''list down 1 important takeaway : {input_data} '''
prompt =  PromptTemplate(template=template,input_variables=['input_data'])

videoSummarizationchain = prompt | llm
result = videoSummarizationchain.invoke({"input_data":youtube_file_data})