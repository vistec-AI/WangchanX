# pip install chromadb==0.5.0 langchain-chroma==0.1.1 langchain-community==0.2.4 openai==0.28.0

import chromadb
from langchain_chroma import Chroma
from langchain_community.embeddings import LocalAIEmbeddings
from langchain_community.document_loaders import CSVLoader

# Load the CSV file
loader = CSVLoader(file_path="./rag.csv")
docs = loader.load()

# Create the client for the Chroma server
client = chromadb.HttpClient(
    host="localhost",
    port=8000,
)

# Create the collection for the documents
collection = client.create_collection("rag-collection")

# Create the embeddings function
embedding_function = LocalAIEmbeddings(
    openai_api_key="random_string",
    openai_api_base="http://localhost:8080",
    model="bge-m3",
)

# Add the documents to the collection
db = Chroma.from_documents(
    client=client,
    collection_name="rag-collection",
    documents=docs,
    embedding=embedding_function,
)
