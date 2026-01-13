import streamlit as st
import pypdf
import os
import torch
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- 1. PAGE CONFIG ---
st.set_page_config(page_title="Smart Study Buddy", page_icon="🎓", layout="wide")

# --- 2. DARK THEME & UI STYLING (CLEANED) ---
st.markdown("""
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.2/css/all.min.css">
    <style>
    .stApp { 
        background-color: #0E1117; 
        color: #FAFAFA; 
    }
    .info-box {
        background-color: #1E2129;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #4CAF50;
        color: #FFFFFF;
        margin: 10px 0;
        line-height: 1.7;
    }
    h1, h2, h3 { color: #4CAF50 !important; }
    .sidebar-link {
        display: flex;
        align-items: center;
        padding: 10px 15px;
        background-color: #262730;
        color: #4CAF50 !important;
        text-decoration: none;
        border-radius: 8px;
        margin: 8px 0;
        font-weight: bold;
    }
    .sidebar-link i {
        margin-right: 12px;
        font-size: 1.2rem;
    }
    .sidebar-link:hover { 
        background-color: #4CAF50; 
        color: white !important; 
    }
    div.stButton > button {
        background-color: #4CAF50 !important;
        color: white !important;
        font-weight: bold !important;
        width: 100% !important;
        height: 3em !important;
        border-radius: 10px !important;
        border: none !important;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 3. LOAD MODELS ---
@st.cache_resource
def load_models():
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    qa_pipe = pipeline("text2text-generation", model="google/flan-t5-large")
    return embedder, qa_pipe

embedder, qa_pipe = load_models()

# --- 4. HELPERS ---
def extract_pdf_text(pdf_file):
    reader = pypdf.PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        content = page.extract_text()
        if content:
            text += content + "\n"
    return text

# --- 5. SIDEBAR ---
with st.sidebar:
    st.header("👤 About Me")
    st.info("Built by J Bahulika")
    
    st.header("🔗 Connect")
    st.markdown(f"""
        <a href="https://www.linkedin.com/in/j-bahulika-8b8237207/" class="sidebar-link" target="_blank">
            <i class="fab fa-linkedin"></i> LinkedIn Profile
        </a>
        <a href="https://github.com/JBahulika" class="sidebar-link" target="_blank">
            <i class="fab fa-github"></i> GitHub Portfolio
        </a>
    """, unsafe_allow_html=True)
    
    st.divider()
    st.header("📁 Upload Center")
    file = st.file_uploader("Upload Notes (PDF)", type="pdf")

# --- 6. MAIN UI ---
st.title("🎓 Smart Study Buddy")

input_method = st.radio("Input Selection:", ["Paste Text", "Upload PDF"], horizontal=True)

user_input = ""
if input_method == "Paste Text":
    user_input = st.text_area("Paste your study material here:", height=250)
else:
    if file:
        user_input = extract_pdf_text(file)
    else:
        st.info("Please upload a PDF file in the sidebar to get started.")

# THE GENERATE BUTTON
if st.button("GENERATE SUMMARY"):
    if not user_input or len(user_input.strip()) < 50:
        st.error("Please provide at least 50 characters of text.")
    else:
        with st.spinner("Analyzing your material..."):
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=100)
            chunks = text_splitter.split_text(user_input)
            
            embeddings = embedder.encode(chunks)
            index = faiss.IndexFlatL2(embeddings.shape[1])
            index.add(np.array(embeddings).astype('float32'))
            
            summary_context = " ".join(chunks[:3]) 
            summary_prompt = f"Summarize these notes clearly: {summary_context}"
            summary_out = qa_pipe(summary_prompt, max_length=512, do_sample=True, repetition_penalty=2.5)[0]['generated_text']
            
            st.session_state['ready'] = True
            st.session_state['summary'] = summary_out
            st.session_state['chunks'] = chunks
            st.session_state['index'] = index

st.divider()

# --- 7. RESULTS AREA ---
if st.session_state.get('ready'):
    col1, col2 = st.columns([1, 1], gap="large")
    
    with col1:
        st.subheader("📝 Detailed Summary")
        st.markdown(f'<div class="info-box">{st.session_state["summary"]}</div>', unsafe_allow_html=True)
    
    with col2:
        st.subheader("🔍 Ask Questions")
        query = st.text_input("Ask a specific question about your document:", placeholder="e.g. What is the main conclusion?")
        
        if query:
            with st.spinner("Searching..."):
                query_vec = embedder.encode([query])
                _, indices = st.session_state['index'].search(np.array(query_vec).astype('float32'), k=2)
                context = " ".join([st.session_state['chunks'][i] for i in indices[0]])
                
                ans_prompt = f"Context: {context}\n\nQuestion: {query}\n\nAnswer:"
                answer = qa_pipe(ans_prompt, max_length=250)[0]['generated_text']
                st.markdown(f'<div class="info-box"><b>AI Buddy:</b><br>{answer}</div>', unsafe_allow_html=True)
else:
    st.write("### 💡 How to use")
    st.write("1. Provide your study notes via paste or PDF upload.")
    st.write("2. Click **Generate** to create a smart summary.")
    st.write("3. Ask specific questions in the chat box to find exact details.")