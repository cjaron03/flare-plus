"""database connection and initialization utilities."""

import logging
from typing import Optional
from contextlib import contextmanager

from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import NullPool

from src.config import DatabaseConfig
from src.data.schema import Base

logger = logging.getLogger(__name__)


class Database:
    """database connection manager."""
    
    def __init__(self, connection_string: Optional[str] = None):
        """
        initialize database connection.
        
        args:
            connection_string: optional sqlalchemy connection string
        """
        self.connection_string = connection_string or DatabaseConfig.get_connection_string()
        self.engine = None
        self.session_factory = None
        
    def connect(self):
        """establish database connection."""
        try:
            self.engine = create_engine(
                self.connection_string,
                echo=False,
                pool_pre_ping=True,  # verify connections before using
                pool_size=5,
                max_overflow=10
            )
            
            # test connection
            with self.engine.connect() as conn:
                conn.execute("SELECT 1")
            
            self.session_factory = sessionmaker(bind=self.engine)
            logger.info("database connection established")
            
        except Exception as e:
            logger.error(f"failed to connect to database: {e}")
            raise
    
    def create_tables(self):
        """create all tables defined in schema."""
        if not self.engine:
            self.connect()
        
        try:
            Base.metadata.create_all(self.engine)
            logger.info("database tables created successfully")
        except Exception as e:
            logger.error(f"failed to create tables: {e}")
            raise
    
    def drop_tables(self):
        """drop all tables (use with caution!)."""
        if not self.engine:
            self.connect()
        
        try:
            Base.metadata.drop_all(self.engine)
            logger.warning("all database tables dropped")
        except Exception as e:
            logger.error(f"failed to drop tables: {e}")
            raise
    
    @contextmanager
    def get_session(self) -> Session:
        """
        context manager for database sessions.
        
        usage:
            with db.get_session() as session:
                session.query(...)
        """
        if not self.session_factory:
            self.connect()
        
        session = self.session_factory()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"database session error: {e}")
            raise
        finally:
            session.close()
    
    def close(self):
        """close database connection."""
        if self.engine:
            self.engine.dispose()
            logger.info("database connection closed")


# global database instance
_db_instance: Optional[Database] = None


def get_database() -> Database:
    """get or create global database instance."""
    global _db_instance
    if _db_instance is None:
        _db_instance = Database()
    return _db_instance


def init_database(drop_existing: bool = False):
    """
    initialize database with tables.
    
    args:
        drop_existing: if true, drop existing tables before creating
    """
    db = get_database()
    db.connect()
    
    if drop_existing:
        db.drop_tables()
    
    db.create_tables()

