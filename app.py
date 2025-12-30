import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

# Config page
st.set_page_config(page_title="Chatbot RAG (Groq)", layout="centered")

st.title("Chatbot RAG - Groq & Llama3")
st.markdown("Responde perguntas baseadas nos PDFs carregados usando Groq (Gratuito/Rápido).")

# Sidebar
with st.sidebar:
    st.header("Configuração")
    
    if "GROQ_API_KEY" in st.secrets:
        api_key = st.secrets["GROQ_API_KEY"]
        st.success("API Key carregada do sistema!", icon="✅")
    else:
        api_key = st.text_input("Groq API Key", type="password")
        st.caption("Obtenha sua chave em: https://console.groq.com/keys")
    
    st.subheader("Arquivos")
    pdf_docs = st.file_uploader("Upload PDFs", accept_multiple_files=True)
    
    processar = st.button("Processar")

# Utils
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def get_conversation_chain(vectorstore, api_key):
    llm = ChatGroq(
        groq_api_key=api_key, 
        model_name="llama3-70b-8192", 
        temperature=0.2
    )
    
    memory = ConversationBufferMemory(
        memory_key='chat_history', 
        return_messages=True
    )
    
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

# Main
if "conversation" not in st.session_state:
    st.session_state.conversation = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = None

if processar and pdf_docs:
    if not api_key:
        st.warning("Insira a Groq API Key para continuar.")
    else:
        with st.spinner("Processando PDFs e criando banco vetorial (Isso pode demorar um pouco na 1ª vez)..."):
            # 1. Extrair Texto
            raw_text = get_pdf_text(pdf_docs)
            
            # 2. Dividir em chunks
            text_chunks = get_text_chunks(raw_text)
            
            # 3. Criar Vetores (Embeddings)
            vectorstore = get_vectorstore(text_chunks)
            
            # 4. Criar Chain
            st.session_state.conversation = get_conversation_chain(vectorstore, api_key)
            
            st.success("Pronto! Pode perguntar.")

# Chat Interface
if st.session_state.conversation:
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    user_question = st.chat_input("Pergunta sobre os documentos:")
    
    if user_question:
        st.session_state.messages.append({"role": "user", "content": user_question})
        with st.chat_message("user"):
            st.markdown(user_question)

        with st.spinner("Llama 3 está pensando..."):
            response = st.session_state.conversation({'question': user_question})
            st.session_state.chat_history = response['chat_history']
            
            bot_response = response['answer']
            
            st.session_state.messages.append({"role": "assistant", "content": bot_response})
            with st.chat_message("assistant"):
                st.markdown(bot_response)

elif not pdf_docs:
    st.info("Carregue PDFs na barra lateral para começar.")
