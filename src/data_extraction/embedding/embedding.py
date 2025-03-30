from langchain.embeddings import HuggingFaceEmbeddings

embedding = HuggingFaceEmbeddings(model="sentence-transformers/all-mpnet-base-v2")
