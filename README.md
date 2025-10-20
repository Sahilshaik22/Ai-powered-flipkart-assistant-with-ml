🤖 AI-Powered Flipkart Policy & Sentiment Assistant

A full-stack AI assistant that answers Flipkart return, refund, and shipping policy queries using LangChain-based RAG, and analyzes customer sentiment from e-commerce reviews using Machine Learning.
Built with Python, Streamlit, LangChain, ChromaDB, Hugging Face embeddings, and OpenAI/Groq APIs.

🚀 Features
🧠 Policy Q&A Assistant (LangChain RAG)

Ingests Flipkart policy PDFs (Return, Refund, Shipping, etc.)

Uses Chroma VectorDB + HuggingFace Embeddings

Retrieves relevant answers via LangChain Retrieval-Augmented Generation (RAG) pipeline

Built-in chat history memory with RunnableWithMessageHistory

💬 Sentiment Analysis Module (ML)

Trained on Indian e-commerce product reviews

Uses TF-IDF + Naive Bayes / Logistic Regression

Classifies reviews into Positive, Negative, or Neutral

Includes visualization and pickle model loading

🖥️ Streamlit Frontend

Interactive multi-mode UI:

📦 Policy Q&A Assistant

🧾 Sentiment Classifier

Clean layout, progress spinners, and error handling



🛠️ Installation & Setup
1️⃣ Install uv (lightweight Python package manager)
pip install uv

2️⃣ Clone this repository
git clone https://github.com/Sahilshaik22/Ai-powered-flipkart-assistant-with-ml.git
cd Ai-powered-flipkart-assistant-with-ml

3️⃣ Create & activate virtual environment
uv venv .venv
uv pip install -r requirements.txt

4️⃣ Set up environment variables

Create a .env file in the root folder:

OPENAI_API_KEY=your_openai_api_key
HF_TOKEN=your_huggingface_token

5️⃣ Vector Database Setup (Optional)

If running for the first time, populate your Vector DB by embedding Flipkart policy PDFs:

python src/ingest_policies.py

This script will:

Load PDFs from Data/raw_policies/

Chunk and embed text using HuggingFace

Store them into VectorDb/

6️⃣ Run the Streamlit App
streamlit run app.py

🧩 Project Structure
Ai-powered-flipkart-assistant-with-ml/
│
├── app.py                        # Streamlit main app
├── src/
│   ├── flipkart_rag_agent.py     # LangChain RAG pipeline
│   ├── ingest_policies.py        # PDF ingestion + embedding
│   ├── nlp_preprocessor.py       # Text cleaning utilities
│   ├── sentiment_model.pkl       # Saved ML model
│
├── Data/
│   ├── raw_policies/             # Flipkart policy PDFs
│   ├── VectorDb/                 # Chroma vector store
│
├── requirements.txt
├── .env.example
└── README.md
