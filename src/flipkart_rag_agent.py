
import os
from typing import Dict
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables import RunnableWithMessageHistory,RunnableLambda
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from dotenv import load_dotenv


load_dotenv() 

os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

class FlipkartRagAgent():
    def __init__(
        self,
        persist_directory:str,
        collection_name:str = "Flipkart_Policies",
        model_name:str = "gpt-4o-mini",
        embedding_model:str = "sentence-transformers/all-MiniLM-L6-v2",
        max_memory_tokens: int = 700
         ):
        
        """
        Args:
            persist_directory: Path to your existing Chroma DB folder.
            collection_name: Name of the Chroma collection.
            model_name: OpenAI chat model name (needs OPENAI_API_KEY).
            embedding_model: HuggingFace sentence embeddings model.
            max_memory_tokens: Reserved for future trimming; not used below.
        """        
   
        self.persist_dir = persist_directory
        self.collection_name = collection_name
        self.model_name = model_name
        self.embedding_model = embedding_model
        self.max_memory_token = max_memory_tokens
        self.retriever = self.load_vectorDb()
        self.llm = self.load_model()
        self._store:dict[str,ChatMessageHistory] = {}
        self.chain_with_history=self.build_chain()
    
    
    def load_vectorDb(self):
        embedding = HuggingFaceEmbeddings(model_name = self.embedding_model)
        vectordb = Chroma(persist_directory=self.persist_dir,embedding_function=embedding,collection_name=self.collection_name)
        retriever = vectordb.as_retriever(
            search_type = "similarity",
            search_kwargs = {"k": 3}
        )
        
        return retriever
    
    def load_model(self):
        llm = ChatOpenAI(model =self.model_name,temperature=0)
        return llm
        
    def build_chain(self):
        contextualize_q_message = (
            "Rewrite the user's question to be self-contained using chat history. "
            "Do NOT answer it.\n\nChat history:\n{chat_history}\n\nUser question: {input}")
        
        contextualize_q_prompt = ChatPromptTemplate.from_messages([
            ("system",contextualize_q_message),
            MessagesPlaceholder("chat_history"),
            ("user","{input}")
        ])
        history_aware_retriever = create_history_aware_retriever(self.llm,self.retriever,contextualize_q_prompt)
        
        system_prompt = (
            "You are Flipkart's Policy Assistant. Answer ONLY using the provided context.\n"
            "If the answer is not found, reply: 'I could not find this in Flipkart policies.'\n\n"
            "Constraints:\n"
            "- Be concise (2-5 sentences)\n"
            "- India-specific context\n"
            "- Reference rules with metadata if available (e.g., source, page)\n\n"
            "Context:\n{context}"
        )
        qa_prompt = ChatPromptTemplate.from_messages([
            ("system",system_prompt),
            MessagesPlaceholder("chat_history"),
            ("user","{input}")
        ])
        

        
        
        question_nd_answer_chain = create_stuff_documents_chain(llm=self.llm,prompt=qa_prompt)
        rag_chain = create_retrieval_chain(history_aware_retriever,question_nd_answer_chain)
        
        to_answer = RunnableLambda(lambda x: x["answer"]) | StrOutputParser()
        final_chain = rag_chain | to_answer
        
        chain_with_history = RunnableWithMessageHistory(final_chain,self._get_message_history,input_messages_key="input",history_messages_key="chat_history")
        

        
        return chain_with_history
    
    

        
    def _get_message_history(self,session_id:str)->ChatMessageHistory:
        if session_id not in self._store:
            self._store[session_id] = ChatMessageHistory()
        return self._store[session_id]
    
    
    
    
    def get_user_query(self,user_query:str,session_id:str = "default"):
        result = self.chain_with_history.invoke(
            {"input":user_query},
            config = {"configurable": {"session_id": session_id}}
        )
        if isinstance(result, dict):
           
           return result.get("answer", str(result))
        return result if isinstance(result, str) else str(result)
    
