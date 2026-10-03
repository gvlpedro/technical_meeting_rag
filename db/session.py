from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from app.config import settings

# NullPool opens a new connection per checkout. It does not reuse pooled connections.
# Tests mix TestClient's event loop with the test function's own loop. A pooled asyncpg
# connection from one loop fails under another loop.
engine = create_async_engine(settings.database_url, poolclass=NullPool)
async_session_factory = async_sessionmaker(engine, expire_on_commit=False)


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    async with async_session_factory() as session:
        yield session
