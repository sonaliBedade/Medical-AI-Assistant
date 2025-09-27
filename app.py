from flask import Flask, render_template, request
from dotenv import load_dotenv
import os

from src.helper import download_embeddings
from src.prompt import *  
from langchain_pinecone import PineconeVectorStore
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain

from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory


# Flask + env
app = Flask(__name__)
load_dotenv()

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY or ""
os.environ["GROQ_API_KEY"] = GROQ_API_KEY or ""

# Embeddings / Vector store / Retriever
embeddings = download_embeddings()

index_name = "virtual-doc"
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings,
)

# A bit more robust than pure similarity
retriever = docsearch.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 4, "fetch_k": 20}
)

# LLM
chatModel = ChatGroq(model="llama-3.3-70b-versatile")

# History-aware question rewriter 
contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "Rewrite the user's follow-up into a standalone question using the chat history. "
     "Only rewrite; do not answer."),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

history_aware_retriever = create_history_aware_retriever(
    chatModel,
    retriever,
    contextualize_q_prompt
)

# Answering prompt that also sees chat history 
answer_prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),             
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

qa_chain = create_stuff_documents_chain(chatModel, answer_prompt)
rag_chain = create_retrieval_chain(history_aware_retriever, qa_chain)

# Session-scoped message history store
SESSION_STORE = {}

def get_session_history(session_id: str) -> ChatMessageHistory:
    if session_id not in SESSION_STORE:
        SESSION_STORE[session_id] = ChatMessageHistory()
    return SESSION_STORE[session_id]

chain_with_history = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)

# Routes
@app.route("/")
def index():
    return render_template("chat.html")

@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    session_id = request.remote_addr or "anon"

    out = chain_with_history.invoke(
        {"input": msg},
        config={"configurable": {"session_id": session_id}}
    )
    return str(out["answer"])

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
