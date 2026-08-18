# 🎓 Smart Study Buddy

**Smart Study Buddy** is an AI-powered educational assistant designed to streamline the learning process. It allows users to upload study notes (PDF) or paste text to instantly generate concise summaries and ask context-aware questions about the material.

🚀 **Live Demo:** [Click here to try the App](https://jbahulika-ssb-project-app-qju3mv.streamlit.app/)

---

## 📜 Project Overview

This project was developed as part of my **IBM Internship** in collaboration with **Edunet Foundation** and **AICTE** (All India Council for Technical Education). The goal was to leverage Generative AI and Natural Language Processing (NLP) to create a functional "Study Buddy" that helps students digest complex materials efficiently.

### Key Features
* **📄 Flexible Input:** Support for uploading PDF documents or pasting raw text directly.
* **📝 Instant Summarization:** Uses the `google/flan-t5-large` model to generate clear, concise summaries of study materials.
* **🔍 Interactive Q&A:** Ask specific questions about your uploaded notes. The system uses RAG (Retrieval-Augmented Generation) to find the exact answer within your document.
* **🎨 Clean UI:** Features a dark-themed, responsive interface built with Streamlit for a focused study environment.

---

## 🛠️ Tech Stack

* **Frontend:** [Streamlit](https://streamlit.io/)
* **LLM & NLP:** * `google/flan-t5-large` (via Hugging Face Transformers)
    * `SentenceTransformers` (all-MiniLM-L6-v2) for embeddings
* **Vector Store:** [FAISS](https://github.com/facebookresearch/faiss) (for efficient similarity search)
* **PDF Processing:** `pypdf`
* **Text Processing:** `LangChain` (RecursiveCharacterTextSplitter)

---

## ⚙️ Installation & Run Locally

To run this project on your local machine, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/JBahulika/Smart-Study-Buddy.git](https://github.com/JBahulika/Smart-Study-Buddy.git)
    cd Smart-Study-Buddy
    ```

2.  **Create a virtual environment (optional but recommended):**
    ```bash
    python -m venv venv
    # On Windows
    venv\Scripts\activate
    # On macOS/Linux
    source venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the application:**
    ```bash
    streamlit run app.py
    ```

---

## 🧠 How It Works

1.  **Ingestion:** The app reads the PDF or text input and splits it into manageable chunks using LangChain's text splitter.
2.  **Embedding:** These chunks are converted into vector embeddings using `SentenceTransformer` and stored in a FAISS index.
3.  **Retrieval:** When you ask a question, the app searches the FAISS index for the most relevant text chunks.
4.  **Generation:** The relevant context is sent to the `Flan-T5` model to generate a natural language response or summary.

---

## 👤 Author

**J Bahulika** *Connect with me:* [![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?style=flat&logo=linkedin)](https://www.linkedin.com/in/j-bahulika-8b8237207/) 
[![GitHub](https://img.shields.io/badge/GitHub-Follow-black?style=flat&logo=github)](https://github.com/JBahulika)

---

*Project developed for the IBM SkillsBuild Internship (Edunet & AICTE).*
