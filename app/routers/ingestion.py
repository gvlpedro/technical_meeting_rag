from datetime import date

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from db.session import get_session
from ingestion.service import NoTranscriptsFoundError, ingest_bronze

router = APIRouter(prefix="/v1", tags=["ingestion"])


class IngestResponse(BaseModel):
    ingestion_date: date
    files_ingested: list[str]
    chunks_created: int


@router.post("/ingest", response_model=IngestResponse)
async def ingest(
    ingestion_date: str, db: AsyncSession = Depends(get_session)
) -> IngestResponse:
    """Ingest every transcript uploaded on `ingestion_date` (compact `YYYYMMDD`, e.g.
    `?ingestion_date=20260906`), looked up at `input/transcriptions/ingestion_date=20260906/`."""
    try:
        result = await ingest_bronze(ingestion_date, db)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid ingestion_date '{ingestion_date}', expected YYYYMMDD",
        )
    except NoTranscriptsFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    return IngestResponse(
        ingestion_date=result.ingestion_date,
        files_ingested=result.files_ingested,
        chunks_created=result.chunks_created,
    )
