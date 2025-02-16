import os
from langchain.chains.question_answering import load_qa_chain
from langchain_openai import AzureChatOpenAI
from keybert.llm import LangChain
from keybert import KeyLLM
from sentence_transformers import SentenceTransformer

os.environ["AZURE_OPENAI_API_KEY"] = "API_KEY"
os.environ["AZURE_OPENAI_ENDPOINT"] = "ENDPOINT"
chain = load_qa_chain(AzureChatOpenAI(
                azure_deployment="gpt-35-turbo",
                openai_api_version="2024-02-15-preview",
            ))
# Create your LLM
llm = LangChain(chain)

# Load it in KeyLLM
kw_model = KeyLLM(llm)

prompt = """
I have the following document:
[DOCUMENT]

Based on the information above, extract the keywords that best describe the topic of the text.
Make sure to only extract keywords that appear in the text.
Use the following format separated by commas:
<keywords>
"""

# Extract keywords
document = ['''I need to determine the optimal time to start and end my work day based on the data provided. This will help me maximize my earnings while minimizing any potential downtime or unnecessary travel.
2. To calculate the optimal start and end times, I need to consider the amount of money earned per hour, the number of orders received per hour, and the total distance traveled in a typical working day.
3. I should also factor in the time it takes to complete each delivery, as well as any additional time spent waiting for pickups or unloading items.
4. Once I have these calculations, I can use them to optimize my work schedule by finding the time when I earn the most money while minimizing any unnecessary travel or downtime.'''
            ,
            "The website mentions that it only takes a couple of days to deliver but I still have not received mine.",
"I received my package!",
"Whereas the most powerful LLMs have generally been accessible only through limited APIs (if at all), Meta released LLaMA's model weights to the research community under a noncommercial license."]

keywords = kw_model.extract_keywords(document, check_vocab=True)
print(keywords)