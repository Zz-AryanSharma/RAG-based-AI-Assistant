import streamlit as st
import os
import datetime
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate

# --- CONFIGURATION ---
st.set_page_config(page_title="IEEE RAS Assistant", page_icon="🤖")

# --- INITIALIZATION ---
# Fallback if API key is not yet set
if "GEMINI_API_KEY" not in st.session_state:
    st.session_state.GEMINI_API_KEY = ""

with st.sidebar:
    st.title("Settings")
    # Hide the API key input from reviewers if it's securely stored in Streamlit Secrets
    if "GEMINI_API_KEY" in st.secrets:
        st.session_state.GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
    else:
        api_key = st.text_input("Gemini API Key (Local Testing)", type="password")
        if api_key:
            st.session_state.GEMINI_API_KEY = api_key

    st.markdown("---")
    st.markdown("Built for **IEEE RAS** using publicly available information.")

    st.markdown("---")
    admin_pw = st.text_input("Admin Panel (Enter Password)", type="password")
    if admin_pw == "Aryan_Sharma":
        st.subheader("Search History Logs")
        if os.path.exists("search_logs.txt"):
            with open("search_logs.txt", "r") as f:
                st.text_area("Reviewer Searches:", f.read(), height=300)
        else:
            st.write("No searches recorded yet.")

# Halt if no API key is available
if not st.session_state.GEMINI_API_KEY:
    st.warning("Please enter your Gemini API Key in the sidebar to continue.")
    st.stop()


@st.cache_resource
def get_vectorstore():
    """Loads documents, or loads the existing vector store if it exists."""
    api_key = st.session_state.GEMINI_API_KEY.strip()
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-2", 
        google_api_key=api_key
    )
    persist_directory = "./chroma_db"
    
    # Check if the database already exists on disk
    if os.path.exists(persist_directory):
        # Load the existing vector store
        vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
        return vectorstore
        
    # Otherwise, load documents and create it
    data_dir = "." 
    
    # Load all .txt files
    loader = DirectoryLoader(data_dir, glob="**/*.txt", loader_cls=TextLoader, show_progress=True)
    docs = loader.load()

    # Split Text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)

    # Create Embeddings and Vector Store, persisting it to disk
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings, persist_directory=persist_directory)
    return vectorstore

from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

def get_rag_chain():
    """Sets up the retrieval chain with LLM and prompt using LCEL."""
    vectorstore = get_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3}) # Retrieve top 3 chunks
    
    api_key = st.session_state.GEMINI_API_KEY.strip()
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash", 
        temperature=0, 
        google_api_key=api_key
    )

    system_prompt = (
        "You are a helpful AI assistant for the IEEE Robotics and Automation Society (RAS). "
        "You were created and programmed by Aryan. If asked who made you, you must state that Aryan created you. "
        "Use the following pieces of retrieved context to answer the question. "
        "If you don't know the answer based on the context, just say that you don't know. "
        "Keep the answer concise and professional.\n\n"
        "Context: {context}"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])
    
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {"context": retriever | format_docs, "input": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain

# --- UI MAIN ---
st.title("IEEE RAS AI Assistant 🤖")
st.write("Ask me anything about IEEE RAS (Membership, Conferences, Technical Committees, etc.) based on the provided documents.")

# Initialize Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("E.g., What are the benefits of membership?"):
    
    # --- Logging Feature ---
    with open("search_logs.txt", "a") as f:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"[{timestamp}] User asked: {prompt}\n")
    # -----------------------

    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    rag_chain = get_rag_chain()

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        with st.spinner("Searching and thinking..."):
            try:
                answer = rag_chain.invoke(prompt)
                st.markdown(answer)
                # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": answer})
            except Exception as e:
                st.error(f"An error occurred: {e}")

