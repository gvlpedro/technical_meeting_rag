from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from app.config import settings

# NullPool: a real connection is opened/closed per checkout instead of reused from a
# pool. Needed because tests mix TestClient's internal event loop with the test
# function's own loop; a pooled asyncpg connection created under one loop can't be
# reused under another ("Task ... attached to a different loop").
engine = create_async_engine(settings.database_url, poolclass=NullPool)
async_session_factory = async_sessionmaker(engine, expire_on_commit=False)


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    async with async_session_factory() as session:
        yield session
