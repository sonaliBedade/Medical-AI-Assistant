from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List
from langchain.schema import Document
import torch
from langchain_groq import ChatGroq
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

def load_file(path):
    loader = DirectoryLoader(path, glob= '*.pdf', loader_cls= PyPDFLoader)
    documents = loader.load()
    return documents

def filtering(documents):
    docs: List[Document] = []
    for doc in documents:
        src= doc.metadata.get("source")
        docs.append(
            Document(
                page_content=doc.page_content,
                metadata={"source":src})
        )
    return docs

def chunking(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size= 1000, chunk_overlap= 200, length_function= len
    )
    text = splitter.split_documents(docs)
    return text

def download_embeddings():
    embeddings= HuggingFaceEmbeddings(
        model_name= "BAAI/bge-small-en-v1.5",
        model_kwargs= {"device":"cuda" if torch.cuda.is_available() else "cpu"}
    )
    return embeddings

