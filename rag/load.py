from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import VectorDBQA

import config
import sys

print("Loading document...")
loader = PyPDFLoader(sys.argv[1])
document = loader.load()

print("Splitting document...")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
texts = text_splitter.split_documents(document)

print("Creating embeddings...")
embedding = OllamaEmbeddings(model=config.AI_MODEL)
vectordb = Chroma.from_documents(documents=texts, embedding=embedding, persist_directory=config.AI_DB_DIR)
