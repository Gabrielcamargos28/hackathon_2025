from http import HTTPStatus

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.config.config import Settings
from src.rag_chain import RAGQueryEngine
from src.schemas.input import InputMessage, OutputMessage

app = FastAPI()
origins = ['*']
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

# Dicionário para armazenar o histórico de mensagens por sessão
chat_histories = {}


@app.post('/', response_model=OutputMessage, status_code=HTTPStatus.OK)
def ask_ia(message: InputMessage):
    session_id = 1
    if session_id not in chat_histories:
        chat_histories[session_id] = []

    def separar_resposta(texto):
        partes = texto.split('<think>')
        if len(partes) > 1:
            pensamento = partes[1].split('</think>')[0].strip()
            resposta_final = texto.split('</think>')[-1].strip()
            return pensamento, resposta_final
        return "", texto.strip()

    GROQ_API_KEY = Settings().GROQ_API_KEY
    CHROMA_DIR = "src/chroma_db"
    query_engine = RAGQueryEngine(groq_api_key=GROQ_API_KEY, persist_directory=CHROMA_DIR)

    # Criar contexto a partir do histórico
    context = "\n".join([f"Usuário: {msg['user']}\nIA: {msg['ia']}" for msg in chat_histories[session_id]])

    question = f"{context}\nUsuário: {message.message}"

    response = query_engine.answer_question(question)
    think, answer = separar_resposta(response['answer'])

    # Adicionar ao histórico
    chat_histories[session_id].append({"user": message.message, "ia": answer})

    return OutputMessage(
        query=message.message,
        think=think,
        answer=answer,
        used_docs=[]
    )
