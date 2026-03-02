from langchain_core.output_parsers import CommaSeparatedListOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_cohere import ChatCohere
from datetime import datetime
from pydantic import BaseModel
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

##CSV Parser -----------------------------------------------------------------------------------------
llm = ChatCohere()

parser = CommaSeparatedListOutputParser()

format_instructions = parser.get_format_instructions()

template = """
List 5 programming languages.

{format_instructions}
"""

prompt = PromptTemplate(
    template=template,
    input_variables=[],
    partial_variables={"format_instructions": format_instructions}
)

chain = prompt | llm | parser

result = chain.invoke({})

print(result)

# Datetime Parser --------------------------------------------------------------------------------------
# Pydantic model for structured output
Server_Logs = [
    "[2024-04-01 13:48:11] ERROR: Failed to connect to database. Retrying in 60 seconds.",
    "[2023-08-04 12:01:00 AM] WARNING: The system is running low on disk space.",
    "[04-01-2024 13:55:39] CRITICAL: System temperature exceeds safe threshold. Initiating shutdown",
    "[Monday, April 01, 2024 01:55:39 PM] DEBUG: User query executed in 0.45 seconds.",
    "[13:55:39 on 2024-04-01] ERROR: Unable to send email notification. SMTP server not responding."
]

# Pydantic model for structured output
class LogDateModel(BaseModel):
    date: datetime

# Initialize parser
parser = PydanticOutputParser(pydantic_object=LogDateModel)

# Prepare template
template = """
Extract the datetime from the following server log line:

{log_line}

{format_instructions}
"""

prompt = PromptTemplate(
    template=template,
    input_variables=["log_line"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

# LLM
llm = ChatCohere()  # or ChatCohere()

chain = prompt | llm | parser

# Process each log line
for log in Server_Logs:
    result = chain.invoke({"log_line": log})
    print(f"Log: {log}")
    print(f"Extracted datetime: {result.date}")
    print("-" * 50)