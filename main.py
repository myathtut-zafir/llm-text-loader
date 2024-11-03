import os
import streamlit as st
import pickle
import time
import langchain
import copy
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.chains.qa_with_sources.retrieval import RetrievalQAWithSourcesChain
from langchain.chains.qa_with_sources.loading import load_qa_with_sources_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import ChatOllama

load_dotenv()

# llm = ChatOpenAI(temperature=0, model_name="gpt-4o-mini", max_tokens=500)
llm = ChatOllama(temperature=0, model="llama3.2", max_tokens=500)
loaders = UnstructuredURLLoader(
    urls=[
        "https://www.moneycontrol.com/news/business/markets/wall-street-rises-as-tesla-soars-on-ai-optimism-11351111.html",
        "https://www.moneycontrol.com/news/business/tata-motors-launches-punch-icng-price-starts-at-rs-7-1-lakh-11098751.html",
    ]
)
data= loaders.load()

text_splitter=RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    )
docs=text_splitter.split_documents(data)

embeddings=OpenAIEmbeddings()
vectorDB=FAISS.from_documents(docs,embeddings)

# simple query - note
# query = "how much Walt Disney (DIS.N) added?"
# docs = vectorDB.similarity_search(query)
# print(docs[0].page_content)
    
chain=RetrievalQAWithSourcesChain.from_llm(llm=llm,retriever=vectorDB.as_retriever())
query = "what is the price of Tiago iCNG?"

langchain.debug=True

print(chain({"question": query}, return_only_outputs=True))



#note
# print(docs[0])
# res = llm.invoke("hello")

# print(res)
