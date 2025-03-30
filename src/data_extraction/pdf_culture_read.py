import os
from glob import glob

from groq import Groq
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from nv_ingest import PDFExtractor

from config.config import Settings


class PDFProcessor:
    def __init__(self, GROQ_API_KEY):
        # Configuração do embedding model
        self.embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/msmarco-distilbert-base-v4"
        )
        # Configuração do cliente Groq
        self.groq_client = Groq(api_key=GROQ_API_KEY)

        # Configuração do text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1200,
            chunk_overlap=300
        )

    def process_pdf(self, path: str) -> None:
        """Processa um único PDF usando NVIDIA Ingest, salva em TXT e armazena embeddings."""
        try:
            print(f"Processando arquivo: {path}")

            # Extrai texto do PDF com NVIDIA Ingest
            extractor = PDFExtractor()
            extracted_text = extractor.extract_text(path)

            if not extracted_text.strip():
                print(f"Nenhum texto extraído do arquivo {path}. Verifique se é um PDF válido.")
                return

            # Salva como TXT
            txt_path = path.replace('.pdf', '.txt')
            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write(extracted_text)

            print(f"Texto salvo em: {txt_path}")

            # Divide o texto em chunks para embeddings
            chunks = self.text_splitter.split_text(extracted_text)

            # Armazena no ChromaDB
            Chroma.from_texts(
                texts=chunks,
                embedding=self.embedding_model,
                persist_directory='src/chroma_db'
            )

            print(f"PDF {path} processado com sucesso!")
        except Exception as e:
            print(f"Erro ao processar {path}: {str(e)}")


if __name__ == '__main__':
    GROQ_API_KEY = Settings().GROQ_API_KEY  # Defina sua chave aqui

    processor = PDFProcessor(GROQ_API_KEY)

    PDF_PATH = 'src/data_extraction/files/'
    all_pdfs = glob(os.path.join(PDF_PATH, '*.pdf'))

    if not all_pdfs:
        print(f"Nenhum PDF encontrado em {PDF_PATH}")
    else:
        for pdf in all_pdfs:
            processor.process_pdf(pdf)
