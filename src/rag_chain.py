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
        """
        Inicializa o motor de consulta RAG para usar com base de dados vetoriais existente.
        
        Args:
            groq_api_key (str): Chave API para o serviço Groq
            persist_directory (str): Diretório onde o ChromaDB está armazenado
        """
        self.persist_directory = persist_directory

        # Verificar se o diretório existe
        if not os.path.exists(persist_directory):
            raise FileNotFoundError(f"O diretório de persistência {persist_directory} não existe")

        logger.info(f"Inicializando motor de consulta RAG usando ChromaDB em {persist_directory}")

        # Configuração do modelo de embedding (deve ser o mesmo usado para criar os embeddings)
        self.embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/msmarco-distilbert-base-v4"
        )

        # Configuração do LLM
        self.llm = ChatGroq(
            api_key=groq_api_key,
            model_name="deepseek-r1-distill-llama-70b",
            temperature=0.1  # Temperatura baixa para respostas mais factuais
        )

        # Carregar o ChromaDB existente
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

        # Criar um retriever a partir do vector store
        retriever = self.vector_store.as_retriever(
            search_type=search_type,
            search_kwargs={"k": k}
        )

        template = """
        Seu nome é AgroMind, você é um assistente especializado em consultoria agrícola no cerrado. Você é um assistente profissional, então sempre responda de modo dissertativo e nunca markdown.
        Portanto deve formular toda a sua resposta em Portugues Brasileiro, inclusive seu pensamento deve sempre ser em português. 
        Sua tarefa é buscar ajuda ao que for pedido no contexto utilizando os documentos inseridos, foque em estrturar suas ideias em paragrafos. 
        Sua especiliazação são 3 culturas: milho, soja e trigo. Só deve responder se forem sobre essas 3 culturas. 
        Sua única fonte de dados são os documentos fornecidos. Caso os documentos fornecidos não sejam suficientes para a resposta, você deve
        falar o que você tem nos documento fornecidos e completar falando que não até o momento essas são as únicas informações que você tem. 
        Sua fonte principal deve ser os documentos, unicamente e exclusivamente os documentos. Você nunca pode usar seu conhecimento prévio para responder a pergunta. 
        Sua resposta deve ser sucinta, utilizando até o máximo de 2500 caracteres.

        {context}
        
        Pergunta: {question}
        
        Resposta:
        """

        # Criar o prompt
        prompt = PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )

        # Configurar a cadeia de QA
        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",  # Método "stuff" concatena todos os documentos
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

            # Extrair fontes únicas para facilitar a referência
            sources = []
            for doc in result.get("source_documents", []):
                source = doc.metadata.get("source", "Desconhecida")
                if source not in sources:
                    sources.append(source)

            # Formatar os resultados
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
        """
        Recupera documentos relevantes do meu banco vetorial.
        Útil para debug ou análise.
        
        Args:
            query (str): A consulta para buscar documentos
            k (int): Número de documentos a recuperar
            
        Returns:
            List: Lista de documentos relevantes
        """
        return self.vector_store.similarity_search(query, k=k)


# Exemplo de uso
if __name__ == "__main__":
    GROQ_API_KEY = Settings().GROQ_API_KEY

    # Diretório onde o ChromaDB está armazenado
    CHROMA_DIR = "chroma_db"

    # Inicializar o motor de consulta
    query_engine = RAGQueryEngine(groq_api_key=GROQ_API_KEY, persist_directory=CHROMA_DIR)

    # Exemplo de consulta
    question = "Identifiquei Nematode Assay em uma amostra de solo. O que devo fazer?"

    # Opção 1: Apenas buscar documentos para depuração/análise
    print("\n=== Documentos relevantes recuperados ===")
    docs = query_engine.get_relevant_documents(question, k=6)
    for i, doc in enumerate(docs):
        print(f"\nDocumento {i + 1} - Fonte: {doc.metadata.get('source', 'Desconhecida')}")
        print(f"Conteúdo: {doc.page_content[:200]}...\n")

    response = query_engine.answer_question(question)
    print(response)

    # print(f"\nPergunta: {response['question']}")
    # print(f"\nResposta: {response['answer']}")

    # print("\nFontes dos documentos:")
    for source in response["sources"]:
        print(f"- {source}")
