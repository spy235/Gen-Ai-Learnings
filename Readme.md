Here's the Markdown code for your LangChain Learning Reference Guide:

# 🚀 LangChain Learning Reference Guide

This repository contains my hands-on learning and practice with **LangChain**, covering LLM integration, chaining, structured outputs, document loaders, and automation workflows.

---

# 🧠 1. Core Concepts Learned

## 🔹 LLM Integrations
- ChatCohere
- ChatOpenAI
- Temperature control (deterministic vs creative output)

---

## 🔹 Prompt Engineering

### PromptTemplate
- Dynamic prompts using variables
- Clean separation of prompt structure and runtime inputs
- Partial variables support

Example:
```python
PromptTemplate(
    input_variables=["theme"],
    template="Provide 10 book titles about {theme}"
)
```

---

## 🔹 LCEL (LangChain Expression Language)

Using `|` operator to build pipelines:

```python
chain = prompt | llm | parser
```

Pipeline flow:

```
Input → Prompt → LLM → Output Parser
```

---

## 🔹 invoke() Method

```python
chain.invoke({"theme": "Finance"})
```

Replaces older `.run()` pattern.

---

# 🔗 2. Chain Types

## ✅ Simple Chain

Single prompt → LLM

## ✅ Multi-Step Sequential Chain

Output of one chain becomes input to another.

Example:

```
Generate Book List → Pick One → Summarize
```

## ✅ Financial Analysis Pipeline

```
Raw Stock Data → Extract Pros/Cons → Investment Advice
```

---

# 📊 3. Output Parsers

## 🔹 StrOutputParser

Returns clean string output.

## 🔹 CommaSeparatedListOutputParser

Forces LLM to return comma-separated values.

## 🔹 PydanticOutputParser

Structured output with validation.

Example:

```python
class EmailResponse(BaseModel):
    Email_Language: str
    Customer_ID: str
    Summary: str
```

Benefits:

- Structured output
- Automatic validation
- Production-ready format control

---

# 📄 4. Document Loaders

All loaders return:

```
List[Document]
```

Each document contains:

- page_content
- metadata

## File Loaders

- TextLoader
- CSVLoader
- PyPDFLoader
- BSHTMLLoader

## Web & External Loaders

- WebBaseLoader
- WikipediaLoader
- YoutubeLoader

---

# 🤖 5. Automation Workflows Built

## ✅ Email Automation System

- Detect language
- Extract customer ID
- Translate to English
- Generate summary
- Generate polite reply
- Structured output using Pydantic

## ✅ Stock Analysis System

- Extract financial metrics
- Identify pros and cons
- Provide investment advice

## ✅ Log Date Extraction

- Extract datetime
- Convert to Python datetime object
- Structured validation

---

# 🏗 6. Architecture Pattern Used

Standard Runnable Pipeline:

```
PromptTemplate
      ↓
LLM
      ↓
Output Parser (Optional)
```

For multi-step systems:

```
Chain 1 → Chain 2 → Chain 3
```

---

# 🔥 7. Skills Acquired

✔ Prompt Engineering  
✔ Structured Output Design  
✔ Schema Validation with Pydantic  
✔ Multi-step LLM Workflows  
✔ Document Ingestion  
✔ Web Scraping via Loaders  
✔ Email Automation  
✔ Financial Reasoning Pipelines  
✔ Deterministic LLM Control  
✔ Sequential Chaining  

