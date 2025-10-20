# app.py
import os
import sys
import pathlib
import time
import streamlit as st
import pickle
import pandas as pd
from pathlib import Path

ROOT = pathlib.Path(__file__).resolve().parent
SRC = ROOT / "src"
if SRC.exists() and str(SRC) not in sys.path:
    sys.path.append(str(SRC))

from nlp_preprocessor import NltkPreprocessor


root = Path(__file__).resolve().parents[1]
vector_dir = root/"VectorDb"

@st.cache_resource(show_spinner=False)
def load_rag_agent():
    from flipkart_rag_agent import FlipkartRagAgent
    return FlipkartRagAgent(
        persist_directory=vector_dir
    )

rag_agent = load_rag_agent()

model_path = r"D:\Genai_Projects\ai_ecom_agent\models\churn_xgb_pipeline.pkl"
with open(model_path, "rb") as f:
    churn_model = pickle.load(f)

model_path_sentiment = r"D:\Genai_Projects\ai_ecom_agent\models\sentiment_pipeline.pkl"
with open(model_path_sentiment, "rb") as f:
    sentiment_model = pickle.load(f)



st.set_page_config(
    page_title="AI-Powered Flipkart Insight Assistant",
    page_icon="🛍️",
    layout="wide"
)
st.title("AI-Powered Flipkart E-Commerce Customer Insight & Retention Assistant")
st.markdown("Analyze customer sentiment, predict churn, and query Flipkart policies — powered by LangChain, RAG, and Machine Learning.")
st.divider()


st.sidebar.title("🧭 Navigation")
app_mode = st.sidebar.radio(
    "Select a Module:",
    ["🏠 Home", "📉 Churn Prediction", "💬 Sentiment Analysis", "🤖 Policy Q&A Assistant"]
)

# Sidebar Footer
st.sidebar.markdown("---")
st.sidebar.caption("Bulit By SAHIL ❤️" )

if app_mode == "🏠 Home":
    st.header("Welcome to Flipkart AI Insight Dashboard 👋")
    st.write("""
    This platform combines Data Science + AI + LangChain RAG to deliver:
    - 💬 Sentiment analysis on product reviews
    - 📉 Customer churn prediction
    - 🤖 AI-powered policy Q&A using Flipkart PDFs
    
    Use the sidebar to explore each feature.
    """)

elif app_mode == "📉 Churn Prediction":
    st.header("📉 Customer Churn Prediction")
    st.write("Upload your customer dataset and get churn probability insights.")
    uploaded_file = st.file_uploader("Upload customer data (CSV)", type="csv")
    if uploaded_file:
        st.success("File uploaded successfully!")
        df_in = pd.read_csv(uploaded_file)
        st.subheader("Preview")
        st.dataframe(df_in.head(10), use_container_width=True)
        st.write("model_predicting....................")
        y_pred = churn_model.predict(df_in)
        df_out = pd.DataFrame(y_pred,columns=["churn"])
        df_concat = pd.concat([df_in,df_out],axis =1 )
        st.write("predicted Table",df_concat)
        churn_counts = df_out["churn"].value_counts().reset_index()
        churn_counts.columns = ["Churn_Status", "Count"]
        churn_counts["Churn_Status"] = churn_counts["Churn_Status"].map({0: "Not Churn", 1: "Churn"})
        st.download_button("⬇️ Download Predictions CSV", data=df_concat.to_csv(index=False).encode('utf-8'), file_name="churn_predictions.csv", mime="text/csv")

        st.subheader("📊 Churn Probability Bar Chart")
        st.bar_chart(churn_counts.set_index("Churn_Status"))
        
        

elif app_mode == "💬 Sentiment Analysis":
    st.header("💬 Product Review Sentiment Analysis")
    uploaded_file = st.file_uploader("Upload customer data (CSV)", type="csv")

    if uploaded_file:
        st.success("File uploaded successfully!")
        df_in = pd.read_csv(uploaded_file)

        st.subheader("Preview")
        st.dataframe(df_in.head(10), use_container_width=True)

        # 1) Pick the text column robustly
        candidate_cols = ["text", "review", "comment", "content", "review_text", "title", "reviews"]
        text_col = next((c for c in df_in.columns if c.lower() in candidate_cols), None)
        if text_col is None:
            st.error("Couldn't find a text column (e.g., 'text' or 'review'). Please rename your text column.")
            st.stop()

        # 2) Clean text (same way used in training)
        cleaner = NltkPreprocessor()
        cleaned = cleaner.transform(df_in[text_col])

        st.write("Model predicting...")
        # 3) Your joblib-loaded pipeline expects a 1D iterable of strings
        y_pred = sentiment_model.predict(cleaned)

        # 4) Show predictions + download
        df_out = pd.DataFrame({"Sentiment": y_pred})
        df_concat = pd.concat([df_in, df_out], axis=1)

        st.subheader("Predicted Table")
        st.dataframe(df_concat.head(20), use_container_width=True)

        st.download_button(
            "⬇️ Download Sentiment Predictions CSV",
            data=df_concat.to_csv(index=False).encode("utf-8"),
            file_name="sentiment_predictions.csv",
            mime="text/csv"
        )

        # 5) Sentiment counts chart
        st.subheader("📊 Sentiment Counts")
        counts = df_out["Sentiment"].value_counts().rename_axis("Sentiment").reset_index(name="Count")
        st.bar_chart(counts.set_index("Sentiment"))

          
        
elif app_mode == "🤖 Policy Q&A Assistant":
    st.header("🤖 Flipkart Policy Q&A Assistant")
    st.write("Ask AI about return, refund, or shipping policies below:")
    

    with st.form("qa_form", clear_on_submit=False):
        user_query = st.text_input("Enter your question:")
        submitted = st.form_submit_button("Get Answer", type="primary")

    if submitted:
        if not user_query.strip():
            st.warning("Please enter a question.")
            st.stop()


        session_id = "session_id"


        with st.spinner("Thinking..."):
            t0 = time.perf_counter()
            try:
                response = rag_agent.get_user_query(
                    user_query=user_query,
                    session_id=session_id,
                )
            except Exception as e:
                st.error(f"Error while querying agent: {e}")
                st.stop()
            latency = time.perf_counter() - t0

        st.subheader("Answer")
        st.success(response if isinstance(response, str) else str(response))
        st.caption(f"⏱️ {latency:.2f} seconds")