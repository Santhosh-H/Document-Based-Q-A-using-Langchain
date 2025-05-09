import os
from langchain_openai import OpenAIEmbeddings,ChatOpenAI
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_retrieval_chain,create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
import streamlit as st
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory


#OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
embeddings=OpenAIEmbeddings(api_key=OPENAI_API_KEY)
llm=ChatOpenAI(model="gpt-4o", api_key=OPENAI_API_KEY)

document = TextLoader("product-data.txt").load()
text_splitter= RecursiveCharacterTextSplitter(chunk_size=1000,
                                              chunk_overlap=200)
chunks=text_splitter.split_documents(document)
vector_store=Chroma.from_documents(chunks,embeddings)
retriever = vector_store.as_retriever()

prompt_template = ChatPromptTemplate.from_messages(
[
    ("system","""You are an assistant for answering questions.
    Use the provided context to respond.If the answer 
    isn't clear, acknowledge that you don't know. 
    Limit your response to three concise sentences.
    {context}
    
    """),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}")
]
)

history_aware_retriever = create_history_aware_retriever(llm,retriever,prompt_template)
qa_chain = create_stuff_documents_chain(llm,prompt_template)
rag_chain = create_retrieval_chain(history_aware_retriever,qa_chain)

history_for_chain = StreamlitChatMessageHistory()

chain_with_history = RunnableWithMessageHistory(
    rag_chain,
    lambda session_id : history_for_chain,
    input_messages_key="input",
    history_messages_key="chat_history"
)

st.write("Chat with Document")
question=st.text_input("Your Question")

if question:
    response = chain_with_history.invoke({"input":question},{"configurable":{"session_id":"abc123"}})
    st.write(response['answer'])

