import streamlit as st
from llama_index.core import StorageContext, load_index_from_storage, Settings
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama

# Set the embedding model to Ollama
Settings.embed_model = OllamaEmbedding(model_name="nomic-embed-text")

# Set the LLM model with a custom temperature (higher temperature = more creative responses)
Settings.llm = Ollama(model="llama3.2", request_timeout=500.0, temperature=0.7)

# Load the documents from storage
storage_context = StorageContext.from_defaults(persist_dir="./storage")
index = load_index_from_storage(storage_context)

# Set up the query engine with additional parameters for better accuracy
query_engine = index.as_query_engine()# Enable verbose mode for detailed responses

# Streamlit app interface
st.title("EDUMATE- AI Powered Study Partner")
st.write("Ask a question based on the loaded documents:")

# User input for the question
question = st.text_input("Enter your question:")

# If a question is entered, perform the query and display the response
if question:
    with st.spinner('Fetching response...'):
        response = query_engine.query(question)

        # Display the response in Streamlit
        st.subheader("Response:")
        st.write(response.response)

        