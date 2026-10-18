"""Structured model-to-Python results. Do not treat schema validity as evidence validity."""
from pydantic import BaseModel, Field

class StatisticalFinding(BaseModel):
    dataset: str
    outcome: str
    predictors: list[str]
    sample_size: int = Field(ge=1)
    coefficients: dict[str,float]
    limitations: list[str]

class LiteratureFinding(BaseModel):
    question: str
    summary: str
    source_urls: list[str]
    limitations: list[str]
