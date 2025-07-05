"""
SQLAlchemy models for the synth_eval database.
This provides a single source of truth for the database schema.
"""

from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    String,
    Float,
    Boolean,
    DateTime,
    JSON,
    ForeignKey,
    Text,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker
from sqlalchemy.sql import func
from typing import List
import os
from pathlib import Path

# Import duckdb_engine to register the dialect
import duckdb_engine

Base = declarative_base()


class Environment(Base):
    __tablename__ = "environments"

    id = Column(Integer, primary_key=True, autoincrement=False)
    name = Column(String, unique=True, nullable=False)  # 'crafter', 'minigrid', etc.
    display_name = Column(String, nullable=False)
    description = Column(Text)
    created_at = Column(DateTime, server_default=func.now())

    # Relationships
    evaluations = relationship("Evaluation", back_populates="environment")


class Evaluation(Base):
    __tablename__ = "evaluations"

    id = Column(Integer, primary_key=True, autoincrement=False)
    env_id = Column(Integer, ForeignKey("environments.id"), nullable=False)
    run_id = Column(String, unique=True, nullable=False)
    timestamp = Column(DateTime, nullable=False)
    models_evaluated = Column(JSON, nullable=False)  # List of model names
    difficulties_evaluated = Column(JSON)  # List of difficulties
    num_trajectories = Column(Integer, nullable=False)
    success_rate = Column(Float)
    avg_achievements = Column(Float)
    eval_metadata = Column("metadata", JSON)
    created_at = Column(DateTime, server_default=func.now())

    # Relationships
    environment = relationship("Environment", back_populates="evaluations")
    trajectories = relationship("Trajectory", back_populates="evaluation")


class Trajectory(Base):
    __tablename__ = "trajectories"

    id = Column(Integer, primary_key=True, autoincrement=False)
    eval_id = Column(Integer, ForeignKey("evaluations.id"), nullable=False)
    trace_id = Column(
        String, unique=True, nullable=False
    )  # e.g., "trace-crafter-eval-001-000"
    model_name = Column(String, nullable=False)
    difficulty = Column(String)
    seed = Column(Integer)
    success = Column(Boolean, nullable=False)
    final_reward = Column(Float)
    num_steps = Column(Integer, nullable=False)
    achievements = Column(JSON)  # List of achievement names
    eval_metadata = Column("metadata", JSON)
    created_at = Column(DateTime, server_default=func.now())

    # Relationships
    evaluation = relationship("Evaluation", back_populates="trajectories")
    trace = relationship("Trace", back_populates="trajectory", uselist=False)


class Trace(Base):
    __tablename__ = "traces"

    id = Column(Integer, primary_key=True, autoincrement=False)
    trajectory_id = Column(Integer, ForeignKey("trajectories.id"), nullable=False)
    parquet_path = Column(String)  # Path to parquet file if stored externally
    trace_format = Column(String, default="parquet")
    size_bytes = Column(Integer)
    created_at = Column(DateTime, server_default=func.now())

    # Relationships
    trajectory = relationship("Trajectory", back_populates="trace")


# Database connection setup
def get_db_url():
    """Get the database URL, using environment variable or default."""
    db_path = os.getenv("TRACE_DB")
    if not db_path:
        # Default to project root
        current_file = Path(__file__).resolve()
        project_root = current_file.parent.parent.parent
        db_path = project_root / "synth_eval.duckdb"
    else:
        db_path = Path(db_path).resolve()

    return f"duckdb:///{db_path}"


def get_engine():
    """Get SQLAlchemy engine for DuckDB."""
    return create_engine(get_db_url())


def get_session():
    """Get SQLAlchemy session."""
    engine = get_engine()
    Session = sessionmaker(bind=engine)
    return Session()


def create_tables():
    """Create all tables if they don't exist."""
    engine = get_engine()
    Base.metadata.create_all(engine)


def drop_tables():
    """Drop all tables."""
    engine = get_engine()
    Base.metadata.drop_all(engine)
