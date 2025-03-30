import os
from glob import glob

from groq import Groq
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain.schema import Document

class PDFProcessor:
    def __init__(self, GROQ_API_KEY, persist_directory='chroma_db'):

        self.embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2"
        )

        self.groq_client = Groq(api_key=GROQ_API_KEY)

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )


        self.persist_directory = persist_directory

        self.vectorstore = Chroma(
            embedding_function=self.embedding_model,
            persist_directory=self.persist_directory
        )

    def process_pdf(self, path: str) -> None:
        """Processa um único PDF e armazena seus embeddings"""
        try:
            print(f"Processando arquivo: {path}")

    
            loader = PyPDFLoader(file_path=path)
            documents = loader.load()

            chunks = self.text_splitter.split_documents(documents)

            self.vectorstore.add_documents(chunks)
            self.vectorstore.persist()

            print(f"PDF {path} processado com sucesso!")
        except Exception as e:
            print(f"Erro ao processar {path}: {str(e)}")

    def process_txt(self, path: str) -> None:
        """Processa um único arquivo TXT e armazena seus embeddings no ChromaDB."""
        try:
            print(f"Processando arquivo TXT: {path}")

            if not os.path.exists(path):
                print(f"Arquivo {path} não encontrado.")
                return

            with open(path, 'r', encoding='utf-8') as f:
                text = f.read()

            documents = [Document(page_content=chunk) for chunk in self.text_splitter.split_text(text)]

            self.vectorstore.add_documents(documents)
            self.vectorstore.persist()

            print(f"TXT {path} processado e salvo no ChromaDB com sucesso!")
        except Exception as e:
            print(f"Erro ao processar {path}: {str(e)}")
