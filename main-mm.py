
import streamlit as st
import langchain
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

llm = ChatOpenAI(temperature=0, model_name="gpt-4o-mini", max_tokens=500)
# llm = ChatOllama(temperature=0, model="llama3.2", max_tokens=500)
loaders = UnstructuredURLLoader(
    urls=[
        "https://burma.irrawaddy.com/article/2024/11/03/392732.html",
        "https://www.bbc.com/burmese/articles/c78dx7jprxlo",
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
    
chain=RetrievalQAWithSourcesChain.from_llm(llm=llm,retriever=vectorDB.as_retriever())
# query = "အမ်းမြို့ နဲ့ အေအေ အကြောင်းသိချင်လို့ပါ?"
# query = "ရခိုင်အကြောင်းသိချင်ပါတယ်?"
query = "ရန်ကုန်အကြောင်းသိချင်ပါတယ်?"

# langchain.debug=True

print(chain({"question": query}, return_only_outputs=True))

