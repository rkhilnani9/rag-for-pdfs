"""
Input/output data classes
"""

from pydantic import BaseModel


class InferenceData(BaseModel):
    query_text: str
    doc_id: str


class InferenceOutput(BaseModel):
    status_code: int
    message: str
    data: str


class CreateEmbeddingData(BaseModel):
    source_file_path: str
    doc_id: str
