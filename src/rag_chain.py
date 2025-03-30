import logging
import os
from typing import Any, Dict

from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq

from src.config.config import Settings

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class RAGQueryEngine:
    def __init__(self, groq_api_key: str, persist_directory: str = "src/chroma_db"):
        self.persist_directory = persist_directory

        if not os.path.exists(persist_directory):
            raise FileNotFoundError(f"O diretório de persistência {persist_directory} não existe")

        logger.info(f"Inicializando motor de consulta RAG usando ChromaDB em {persist_directory}")

        self.embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/msmarco-distilbert-base-v4"
        )

        self.llm = ChatGroq(
            api_key=groq_api_key,
            model_name="deepseek-r1-distill-llama-70b",
            temperature=0.1
        )

        try:
            self.vector_store = Chroma(
                persist_directory=persist_directory,
                embedding_function=self.embedding_model
            )
            collection_size = len(self.vector_store._collection.get()['ids'])
            logger.info(f"ChromaDB carregado com sucesso. Coleção contém {collection_size} documentos.")
        except Exception as e:
            logger.error(f"Erro ao carregar ChromaDB: {str(e)}")
            raise

    def setup_qa_chain(self, k: int = 4, search_type: str = "similarity") -> RetrievalQA:
        retriever = self.vector_store.as_retriever(
            search_type=search_type,
            search_kwargs={"k": k}
        )

        template = """
        RESTRIÇÕES: Responder apenas perguntas relacionaadas ao agro. 
        Entre nossas culturas milho, soja e trigo. 
        Suas pragas. Plantio e biodefencivos. Apenas texto limpo.
        SEU PAPEL: Você é AgroMind, um assistente virtual brasileiro especializado em consultoria agrícola no cerrado. Portanto todas suas respostas e pensamentos devem ser em português brasileiro. Você deve ser um assistente profissional texto limpo.
        SUA AÇÃO: Você deve responder a pergunta do usuário ```{question}```. Use seu conhecimento para entender a pergunta e use seus documentos para responder.
        SEU CONTEXTO:
        1. Nunca aceite palavras torpes do usuário.
        2. Suas respostas devem vir exclusivamente dos documentos fornecidos no embedding.
        3. Caso os documentos inseridos não sejam suficientes para responder a perguntam responda com o que tem de informação, sem usar seu conhecimento prévio. Contexto adicional: ```{context}```
        FORMA: Sua resposta nunca, deve passar de 2000 caracteres.
        """

        prompt = PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )

        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt}
        )

        return qa_chain

    def answer_question(self, question: str, k: int = 4) -> Dict[str, Any]:
        logger.info(f"Processando pergunta: {question}")
        qa_chain = self.setup_qa_chain(k=k)

        try:
            result = qa_chain({"query": question})

            sources = []
            for doc in result.get("source_documents", []):
                source = doc.metadata.get("source", "Desconhecida")
                if source not in sources:
                    sources.append(source)

            response = {
                "question": question,
                "answer": result["result"],
                "source_documents": result["source_documents"],
                "sources": sources
            }

            logger.info(f"Resposta gerada com sucesso usando {len(result['source_documents'])} documentos")
            return response

        except Exception as e:
            logger.error(f"Erro ao processar a pergunta: {str(e)}")
            return {
                "question": question,
                "answer": f"Ocorreu um erro ao processar sua pergunta: {str(e)}",
                "source_documents": [],
                "sources": []
            }

    def get_relevant_documents(self, query: str, k: int = 5):
        return self.vector_store.similarity_search(query, k=k)


if __name__ == "__main__":
    GROQ_API_KEY = Settings().GROQ_API_KEY
    CHROMA_DIR = "chroma_db"
    
    query_engine = RAGQueryEngine(groq_api_key=GROQ_API_KEY, persist_directory=CHROMA_DIR)
    
    # question = "Identifiquei Nematode Assay em uma amostra de solo. O que devo fazer?"
    question = "Qual o limite de caracteres que voce pode responder?"
    
    docs = query_engine.get_relevant_documents(question, k=6)
    for i, doc in enumerate(docs):
        print(f"\nDocumento {i + 1} - Fonte: {doc.metadata.get('source', 'Desconhecida')}")
        print(f"Conteúdo: {doc.page_content[:200]}...\n")
    
    response = query_engine.answer_question(question)
    print(f"\nPergunta: {response['question']}")
    print(f"\nResposta: {response['answer']}")
    
    print("\nFontes dos documentos:")
    for source in response["sources"]:
        print(f"- {source}")
    
    print("\n=== Documentos relevantes utilizados ===")
    for i, doc in enumerate(response["source_documents"]):
        print(f"\nDocumento {i + 1} - Fonte: {doc.metadata.get('source', 'Desconhecida')}")
        print(f"Conteúdo: {doc.page_content[:200]}...\n")