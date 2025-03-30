import os
from glob import glob
from typing import List

import pandas as pd
from groq import Groq
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma  # Importação atualizada
from langchain_huggingface import (
    HuggingFaceEmbeddings,  # Importação atualizada
)


class CSVProcessor:
    def __init__(self, GROQ_API_KEY: str):
        # Configuração do embedding model atualizado
        self.embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/msmarco-distilbert-base-v4"
        )
        self.GROG_API_KEY = GROQ_API_KEY

        # Configuração do cliente Groq
        self.groq_client = Groq(api_key=GROQ_API_KEY)

        # Configuração do text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

    def _convert_df_to_documents(self, df: pd.DataFrame, source: str) -> List[Document]:
        """Converte um DataFrame em uma lista de Documentos do LangChain"""
        documents = []
        for _, row in df.iterrows():
            content = "\n".join(f"{col}: {val}" for col, val in row.items() if pd.notna(val))
            documents.append(
                Document(
                    page_content=content,
                    metadata={
                        "source": source,
                        "row_index": _,
                        "columns": list(row.index)
                    }
                )
            )
        return documents

    def process_csv(self, path: str) -> None:
        """Processa um único CSV e armazena seus embeddings"""
        try:
            print(f"Processando arquivo: {path}")

            # Carrega o CSV usando pandas para maior flexibilidade
            df = pd.read_csv(path, delimiter=';', encoding='ISO-8859-1')

            # Converte para documentos LangChain
            documents = self._convert_df_to_documents(df, os.path.basename(path))

            # Divide o texto em chunks
            chunks = self.text_splitter.split_documents(documents)

            # Armazena no ChromaDB
            Chroma.from_documents(
                documents=chunks,
                embedding=self.embedding_model,
                persist_directory='src/chroma_db_csv',
                collection_metadata={"hnsw:space": "cosine"}  # Otimização para similaridade
            )

            print(f"CSV {path} processado com sucesso!")
        except Exception as e:
            print(f"Erro ao processar {path}: {str(e)}")

    def query_groq(self, question: str) -> str:
        """Envia uma consulta para a API da Groq"""
        try:
            response = self.groq_client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": "Você é um analista de dados especializado. Forneça insights claros e acionáveis."
                    },
                    {
                        "role": "user",
                        "content": question
                    }
                ],
                model="deepseek-r1-distill-llama-70b",
                temperature=0.3  # Menos criativo, mais factual
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Erro na consulta à Groq: {str(e)}"


if __name__ == '__main__':
    # Configuração
    GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY não encontrada nas variáveis de ambiente")

    # Cria o processador
    processor = CSVProcessor(GROQ_API_KEY)

    # Processa todos os CSVs no diretório
    CSV_PATH = 'src/data_extraction/files/'
    all_csvs = glob(os.path.join(CSV_PATH, '*.csv'))

    if not all_csvs:
        print(f"Nenhum CSV encontrado em {CSV_PATH}")
    else:
        for csv_file in all_csvs:
            processor.process_csv(csv_file)

    # Exemplo de consulta analítica
    resposta = processor.query_groq(
        "Baseado nos dados processados, quais são as 3 principais tendências ou padrões observáveis? "
        "Forneça uma análise concisa com exemplos específicos quando possível."
    )
    print("\nAnálise dos Dados:")
    print(resposta)
