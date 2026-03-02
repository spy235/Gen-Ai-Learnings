from langchain_cohere import ChatCohere
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

SBIN_Stock_Analysis = """
Company name is State Bank of India
NSE Symbol is SBIN
MARKET CAP - ₹ 6,69,078.16 Cr.
Company has a good Return on Equity (ROE) track record: 3 Years ROE 13.46%.
CASA stands at 42.67% of total deposits.
The company has delivered good Profit growth of 51.35% over the past 3 years.
Company has delivered good profit growth of 76.1% CAGR over last 5 years.
Company has been maintaining a healthy dividend payout of 17.3%.
Company's working capital requirements have reduced from 152 days to 118 days
The bank has a very low ROA track record. Average ROA of 3 years is 0.70%.
Low other Income proportion of 11.03%.High Cost to income ratio of 53.87%.
Company has low interest coverage ratio.
The company has delivered a poor sales growth of 8.91% over past five years.
Company has a low return on equity of 12.8% over last 3 years.
Contingent liabilities of Rs.19,00,096 Cr.
Company might be capitalizing the interest cost.
Earnings include an other income of Rs.1,39,611 Cr.
"""
from langchain_cohere import ChatCohere
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

llm = ChatCohere(temperature=0)

# -------- First Prompt --------
template1 = """
Read the text data from {stock_analysis_input}.
Mention the company name and market capital.
Write top 3 positive and top 3 negative points.
Keep the points short.
"""

stock_analysis_prompt = PromptTemplate(
    input_variables=["stock_analysis_input"],
    template=template1
)

stock_analysis_chain = stock_analysis_prompt | llm | StrOutputParser()

# -------- Second Prompt --------
template2 = """
Imagine you've been analyzing stocks for over 15 years.
Look at the good and bad points below and see if the company can grow.

Right now, is buying shares of this company a smart move?

Pros and Cons:
{Pros_and_Cons}
"""

stock_pick_prompt = PromptTemplate(
    input_variables=["Pros_and_Cons"],
    template=template2
)

stock_pick_chain = stock_pick_prompt | llm | StrOutputParser()

# -------- Sequential Chain --------
stock_chain = stock_analysis_chain | stock_pick_chain

# -------- Run --------
result = stock_chain.invoke({
    "stock_analysis_input": SBIN_Stock_Analysis
})

print(result)