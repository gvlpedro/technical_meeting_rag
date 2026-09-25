from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from app.config import settings

# We use NullPool. This opens and closes a real connection on every checkout instead of
# reusing one from a pool. We need this because tests mix TestClient's own event loop
# with the test function's loop. A pooled asyncpg connection made under one loop cannot
# be reused under another loop.
engine = create_async_engine(settings.database_url, poolclass=NullPool)
async_session_factory = async_sessionmaker(engine, expire_on_commit=False)


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    async with async_session_factory() as session:
        yield session
