import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import pandas as pd
import datetime
import os

# --- UI Setup ---
st.set_page_config(page_title="Expert AI Research Agent", layout="wide")
st.title("🤖 Multi-Tool Research Agent")
st.markdown("### Advanced RAG + Automated Data Logging")

# --- 1. Knowledge Base (Local RAG) ---
@st.cache_resource
def process_pdfs(uploaded_files):
    all_text = []
    for file in uploaded_files:
        # Save temp file to read
        with open("temp.pdf", "wb") as f:
            f.write(file.getbuffer())
        loader = PyPDFLoader("temp.pdf")
        all_text.extend(loader.load())
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(all_text)
    
    # Use FREE local embeddings (No API Key needed)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_db = FAISS.from_documents(chunks, embeddings)
    return vector_db

# --- UI Sidebar ---
with st.sidebar:
    st.header("📁 Upload Documents")
    files = st.file_uploader("Upload Company PDFs", accept_multiple_files=True)
    if files:
        with st.spinner("Indexing documents..."):
            db = process_pdfs(files)
        st.success("Documents Indexed!")

# --- 2. The Agent Logic ---
query = st.text_input("Ask the Agent to analyze something:")

if query and files:
    # Search for relevant info
    results = db.similarity_search(query, k=3)
    context = "\n".join([doc.page_content for doc in results])
    
    # Show the "Thinking" process
    with st.expander("👁️ View Agent Reasoning"):
        st.write("1. Searching local vector database...")
        st.write(f"2. Found {len(results)} relevant segments.")
        st.write("3. Synthesizing data for logging...")

    # Final Response Display
    st.subheader("Final Analysis")
    st.info(f"**Context found in Documents:**\n\n{context[:1000]}...")
    
    # --- 3. The "Action" (Automated Logging) ---
    st.divider()
    if st.button("Confirm & Log to System"):
        log_filename = "audit_log.csv"
        log_entry = {
            "Date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
            "Query": query,
            "Verdict": "Analysis Verified",
            "Source": files[0].name
        }
        df = pd.DataFrame([log_entry])
        
        # EXPERT MODIFICATION: Check if file exists to avoid header errors
        file_exists = os.path.isfile(log_filename)
        df.to_csv(log_filename, mode='a', index=False, header=not file_exists)
        
        st.success(f"✅ Entry added to `{log_filename}` successfully!")
