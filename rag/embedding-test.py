from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma

import config
import time

def parse_results(results):
    for i, result in enumerate(results):
        print(f"{i}: {result.page_content}")
        print(f"{i}: page: {result.metadata['page_number']}")

embedding = OllamaEmbeddings(model=config.AI_EMBEDDING_MODEL)

vectordb = Chroma(persist_directory=config.AI_DB_DIR, embedding_function=embedding)

query = input(">>")

# default K is 4 for all calls
print("Similarity search:")
results = vectordb.similarity_search(query, k=4)
parse_results(results)

# it looks like retriever getting hasn't got k parameter to pass
# TODO: search a way to retrieve more results
retriever = vectordb.as_retriever()
print("Similarity search with retriever:")
results = retriever.get_relevant_documents(query)
parse_results(results)

retriever = vectordb.as_retriever(search_type="mmr")
print("MMR search with retriever:")
results = retriever.get_relevant_documents(query)
parse_results(results)

# let's use LLM to get keyword from the question and use it in search
# instead of using whole input
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms import Ollama
from operator import itemgetter

model = Ollama(model=config.AI_LLM_MODEL)
template = """Make this sentance better for searching in technical documentation.
Don't add any explanation. Don't try to answer the question.

{text}
"""
parser = StrOutputParser()
prompt = ChatPromptTemplate.from_template(template)
chain = ({"text" : itemgetter("text")} | prompt | model | parser)
new_query = chain.invoke({"text": query})
print(new_query)

print("Similarity search with new query:")
results = vectordb.similarity_search(new_query, k=4)
parse_results(results)

