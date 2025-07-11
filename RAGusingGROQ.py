import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

groq_api_key = os.getenv('GROQ_API_KEY')

from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
import streamlit as st
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_core.messages import HumanMessage,AIMessage
from langchain_core.stores import InMemoryStore

model = ChatGroq(model='gemma2-9b-it',groq_api_key=groq_api_key)
model

prompt = ChatPromptTemplate.from_template(
    """
    Answer the following questions based on the given context only.
    <context>
    {context}
    <context>
    Question : {input}
    """
)

def create_vector_embedding():
    st.session_state.loader = PyPDFLoader('maternal_health2.pdf')
    st.session_state.docs = st.session_state.loader.load()
    st.session_state.embeddings = GoogleGenerativeAIEmbeddings(model='models/embedding-001')
    st.session_state.splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
    st.session_state.final_documents = st.session_state.splitter.split_documents(st.session_state.docs)

    st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents,st.session_state.embeddings)


st.title("Maternal HealthCare Chatbot")
user_prompt = st.text_input("Ask anything to the CHATBOT") 

if st.button("Embed Documents"):
    create_vector_embedding()
    st.write("Vector DB is ready")

if user_prompt:
    document_chain = create_stuff_documents_chain(model,prompt)
    retriever = st.session_state.vectors.as_retriever()
    rag_chain = create_retrieval_chain(retriever,document_chain)

    import time
    start = time.process_time()
    response = rag_chain.invoke({'input': user_prompt})
    # print(f"Response time = {time.process_time()-start}")

    st.write(response['answer'])
