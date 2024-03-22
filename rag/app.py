from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory

from langchain.memory import ConversationBufferMemory

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.output_parsers import StrOutputParser

from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks.manager import CallbackManager

from operator import itemgetter

import config
import time

callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
model = Ollama(model=config.AI_MODEL, callbacks=callback_manager)

prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """Answer the question based on the context below. If you can't
                answer the question, replay "I don't know"
                Context: {context}"""
            ),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{question}"),
        ])

embedding = OllamaEmbeddings(model=config.AI_MODEL)
parser = StrOutputParser()

vectordb = Chroma(persist_directory=config.AI_DB_DIR, embedding_function=embedding)
retriever = vectordb.as_retriever()

chain = (
        {
            "history" : itemgetter("history"),
            "context" : itemgetter("question") | retriever,
            "question" : itemgetter("question"),
        }
        | prompt
        | model
        | parser)

store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

message_history = RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="question",
        history_messages_key="history")

while (True):
    question = input(">>")
    if question.startswith("/"):
        break

    start = time.time()
    answer = message_history.invoke({"question": question},
                                    {"configurable": {"session_id": "ai_fun"}})
    exe_time = time.time() - start
    print(f"\n\tExecution time: {exe_time:.6f} seconds")
