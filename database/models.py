from sqlalchemy import Column, String, Integer, DateTime, JSON, Float
from sqlalchemy.sql import func
from .neon import Base

class Document(Base):
    __tablename__ = "documents"

    id = Column(String, primary_key=True, index=True)
    filename = Column(String, index=True)
    modality = Column(String)  # pdf, image, audio
    status = Column(String)    # uploaded, processing, completed, failed
    collection_id = Column(String, index=True)
    b2_file_key = Column(String)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

class Job(Base):
    __tablename__ = "jobs"

    id = Column(String, primary_key=True, index=True)
    document_id = Column(String, index=True)
    status = Column(String)
    progress = Column(Integer, default=0)
    error = Column(String, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
