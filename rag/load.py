from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import UnstructuredFileLoader

from langchain_community.vectorstores import utils as chromautils
from langchain_community.vectorstores import Chroma

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import VectorDBQA

import config
import time
import sys

print("Loading document...")
start = time.time()
loader = UnstructuredFileLoader(sys.argv[1], mode="elements")
document = loader.load()
document = chromautils.filter_complex_metadata(document)
exe_time = time.time() - start
print(f"Loading time: {exe_time:.6f} seconds")

print("Splitting document...")
start = time.time()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
texts = text_splitter.split_documents(document)
exe_time = time.time() - start
print(f"Splitting time: {exe_time:.6f} seconds")

print("Creating embeddings...")
start = time.time()
embedding = OllamaEmbeddings(model=config.AI_MODEL)
vectordb = Chroma.from_documents(documents=texts, embedding=embedding, persist_directory=config.AI_DB_DIR)
exe_time = time.time() - start
print(f"Creating embeddings time: {exe_time:.6f} seconds")
