
from pydantic import BaseModel


class InputMessage(BaseModel):
    session_id: int
    message: str


class DocumentInfo(BaseModel):
    author: str
    total_pages: int
    page: int


# Modelo principal para a mensagem de saída
class OutputMessage(BaseModel):
    query: str
    think: str
    answer: str
