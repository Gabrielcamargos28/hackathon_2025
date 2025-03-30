import os
from glob import glob

from groq import Groq
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

class PDFProcessor:
    def __init__(self, GROQ_API_KEY):
        # Configuração do embedding model
        self.embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2"
        )

        # Configuração do cliente Groq
        self.groq_client = Groq(api_key=GROQ_API_KEY)

        # Configuração do text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,  # Tamanho aumentado para melhorar a qualidade dos embeddings
            chunk_overlap=200  # Overlap aumentado para manter contexto
        )

    def process_pdf(self, path: str) -> None:
        """Processa um único PDF e armazena seus embeddings"""
        try:
            print(f"Processando arquivo: {path}")

            # Carrega o documento PDF
            loader = PyPDFLoader(file_path=path)
            documents = loader.load()

            # Divide o texto em chunks
            chunks = self.text_splitter.split_documents(documents)

            # Armazena no ChromaDB
            Chroma.from_documents(
                documents=chunks,
                embedding=self.embedding_model,
                persist_directory='chroma_db'
            )

            print(f"PDF {path} processado com sucesso!")
        except Exception as e:
            print(f"Erro ao processar {path}: {str(e)}")

    def process_txt(self, path: str) -> None:
            """Processa um único arquivo TXT e armazena seus embeddings no ChromaDB."""
            try:
                print(f"Processando arquivo TXT: {path}")

                # Verifica se o arquivo existe
                if not os.path.exists(path):
                    print(f"Arquivo {path} não encontrado.")
                    return

                # Lê o conteúdo do arquivo TXT
                with open(path, 'r', encoding='utf-8') as f:
                    text = f.read()

                # Divide o texto em chunks
                documents = [Document(page_content=chunk) for chunk in self.text_splitter.split_text(text)]

                # Armazena no ChromaDB
                Chroma.from_documents(
                    documents=documents,
                    embedding=self.embedding_model,
                    persist_directory='chroma_db'
                )

                print(f"TXT {path} processado e salvo no ChromaDB com sucesso!")
            except Exception as e:
                print(f"Erro ao processar {path}: {str(e)}")

def query_groq(self, question: str) -> str:
        """Envia uma consulta para a API da Groq"""
        try:
            response = self.groq_client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "Você é um assistente especialista em documentos."},
                    {"role": "user", "content": question}
                ],
                model="deepseek-r1-distill-llama-70b",
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Erro na consulta à Groq: {str(e)}"


if __name__ == '__main__':
    # Cria o processador
    processor = PDFProcessor()

    # Processa todos os PDFs no diretório
    PDF_PATH = 'src/data_extraction/files/'
    all_pdfs = glob(os.path.join(PDF_PATH, '*.pdf'))

    if not all_pdfs:
        print(f"Nenhum PDF encontrado em {PDF_PATH}")
    else:
        for pdf in all_pdfs:
            processor.process_pdf(pdf)