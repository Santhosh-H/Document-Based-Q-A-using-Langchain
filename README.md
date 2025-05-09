# Document-Based-Q-A-using-Langchain
📄 Document-Based Q&amp;A Chatbot with Streamlit and LangChain

This project is a RAG (Retrieval-Augmented Generation) chatbot interface that allows users to ask questions about the content of a text file. 
The system integrates OpenAI's GPT-4o via LangChain, stores document embeddings in Chroma vector DB, and supports conversational memory using Streamlit chat history.

🧠 Key Design Choices
LangChain RAG Pattern: Combines vector retrieval (Chroma) with GPT-4o response generation.

History-Aware Retriever: Maintains context between user messages for better continuity.

Concise Prompt Engineering: Restricts the assistant to respond within three sentences for clarity and focus.

Streamlit UI: Provides an easy-to-use frontend for interacting with the chatbot.

Chat Memory: Uses StreamlitChatMessageHistory to retain conversational history across user inputs.

