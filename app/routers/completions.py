from typing import Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from llm.router import AllProvidersFailedError, complete

router = APIRouter(prefix="/v1", tags=["completions"])


class CompletionMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class CompletionRequest(BaseModel):
    messages: list[CompletionMessage]


class CompletionResponse(BaseModel):
    content: str
    model: str


@router.post("/completions", response_model=CompletionResponse)
async def create_completion(request: CompletionRequest) -> CompletionResponse:
    messages = [message.model_dump() for message in request.messages]
    try:
        response = await complete(messages)
    except AllProvidersFailedError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    return CompletionResponse(content=response.choices[0].message.content, model=response.model)
