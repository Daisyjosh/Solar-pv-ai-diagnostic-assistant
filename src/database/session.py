"""SQLAlchemy database engine, session factory, and lifecycle helpers."""

from typing import Generator

from sqlalchemy import create_engine, event, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import declarative_base, sessionmaker

from src.core.config import get_settings

settings = get_settings()

# Configure SQLite-specific connection arguments if SQLite is used
connect_args = {}
if settings.database_url.startswith("sqlite"):
    connect_args = {"check_same_thread": False}

engine = create_engine(
    settings.database_url,
    connect_args=connect_args,
    echo=settings.debug and settings.app_env == "development",
)


# Enforce SQLite foreign key integrity constraints
@event.listens_for(Engine, "connect")
def set_sqlite_pragma(dbapi_connection, connection_record):
    """Enable SQLite foreign key support on new database connections."""
    if settings.database_url.startswith("sqlite"):
        cursor = dbapi_connection.cursor()
        try:
            cursor.execute("PRAGMA foreign_keys=ON")
        except Exception:
            pass
        finally:
            cursor.close()


SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


def get_db() -> Generator:
    """Yield a database session and safely close it after request completion."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    """Initialize database tables registered in Base metadata."""
    # Import all models to ensure metadata registration before create_all
    import src.database.models  # noqa: F401

    Base.metadata.create_all(bind=engine)


def check_db_health() -> bool:
    """Execute lightweight query to verify live database connectivity."""
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception:
        return False
