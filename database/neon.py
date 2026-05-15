import os
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy.orm import declarative_base
from dotenv import load_dotenv, find_dotenv

# Try to find .env or .env.local in parent directories
env_path = find_dotenv(".env.local") or find_dotenv(".env")
load_dotenv(env_path)

# We use asyncpg for Neon Postgres
# The URL should start with postgresql+asyncpg://
DATABASE_URL = os.getenv("POSTGRES_URL")
if DATABASE_URL:
    if DATABASE_URL.startswith("postgresql://"):
        DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://", 1)
    if "sslmode=require" in DATABASE_URL:
        DATABASE_URL = DATABASE_URL.replace("sslmode=require", "ssl=require")
    # asyncpg does not support channel_binding
    if "channel_binding=require" in DATABASE_URL:
        DATABASE_URL = DATABASE_URL.replace("&channel_binding=require", "")
        DATABASE_URL = DATABASE_URL.replace("?channel_binding=require", "")

engine = None
if DATABASE_URL:
    engine = create_async_engine(
        DATABASE_URL,
        echo=False,
        pool_size=5,
        max_overflow=10,
        pool_pre_ping=True,
        pool_recycle=300
    )

AsyncSessionLocal = None
if engine:
    AsyncSessionLocal = async_sessionmaker(
        engine, class_=AsyncSession, expire_on_commit=False
    )

Base = declarative_base()

async def get_db():
    if not AsyncSessionLocal:
        raise Exception("Database session local not initialized. Check POSTGRES_URL.")
    async with AsyncSessionLocal() as session:
        yield session
