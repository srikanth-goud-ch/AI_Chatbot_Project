import streamlit as st
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# 1. Page Configuration
st.set_page_config(page_title="Srikanth's AI Doc Assistant", layout="centered")
st.title("📄 AI Documentation Assistant")
st.write("Upload a technical manual and ask questions in real-time.")

load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

# 2. Sidebar for File Upload
with st.sidebar:
    uploaded_file = st.file_uploader("Upload your PDF", type="pdf")
    if uploaded_file:
        with open("temp.pdf", "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success("File uploaded successfully!")

# 3. The "Brain" Logic (Only runs if a file is uploaded)
if uploaded_file and api_key:
    # Use a "Spinner" so the user knows the AI is thinking
    with st.spinner("Analyzing document..."):
        loader = PyPDFLoader("temp.pdf")
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(docs)
        
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
        
        llm = ChatGroq(model="llama-3.1-8b-instant", groq_api_key=api_key)
        
        system_prompt = "You are a helpful assistant. Use the context to answer: \n\n {context}"
        prompt = ChatPromptTemplate.from_messages([("system", system_prompt), ("human", "{input}")])
        
        combine_docs_chain = create_stuff_documents_chain(llm, prompt)
        rag_chain = create_retrieval_chain(vectorstore.as_retriever(), combine_docs_chain)

    # 4. The Chat Interface
    user_query = st.chat_input("Ask something about your document...")
    if user_query:
        st.chat_message("user").write(user_query)
        response = rag_chain.invoke({"input": user_query})
        with st.chat_message("assistant"):
            st.write(response["answer"])
else:
    st.info("Please upload a PDF to get started.")