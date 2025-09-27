# eval_basic.py (no-reference run)
from dotenv import load_dotenv
load_dotenv()

from src.helper import download_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_groq import ChatGroq
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from src.prompt import system_prompt

from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy
from datasets import Dataset

# 1) rebuild your RAG pipeline
embeddings = download_embeddings()
docsearch = PineconeVectorStore.from_existing_index(
    index_name="virtual-doc",
    embedding=embeddings
)
retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.1)
prompt = ChatPromptTemplate.from_messages(
    [("system", system_prompt), ("human", "{input}")]
)
qa_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, qa_chain)

# 2) questions
questions = [
    "Why is ultrasound considered safer than x-ray imaging for the abdomen?",
    "Does achalasia typically reduce life expectancy?",
    "What is the difference between septic and sterile abscesses?",
    "Give two prevention tips to minimize acne flare-ups.",
    "What is stereotactic radiation (radiosurgery) for acoustic neuroma?",
    "What is the general prognosis for someone with a unilateral acoustic neuroma?",
    "Why can abscesses of the hand be especially serious?",
    "What is the first-line treatment for achalasia?",
    "What is the most common symptom of achalasia?",
    "I got sudden fever, what should I do?"
]

# 3) collect contexts + answers (use retriever.invoke)
rows = []
for q in questions:
    ctx_docs = retriever.invoke(q)                 # <- new API (list[Document])
    ctx_texts = [d.page_content for d in ctx_docs]
    out = rag_chain.invoke({"input": q})
    rows.append({"question": q, "answer": out["answer"], "contexts": ctx_texts})

ds = Dataset.from_list(rows)

# 4) evaluate metrics that don't need references
result = evaluate(
    ds,
    metrics=[faithfulness, answer_relevancy],
    llm=llm,
    embeddings=embeddings
)

print("\n=== RAGAS SUMMARY (no-reference metrics) ===")
print(result)