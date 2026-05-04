"""
api/schemas.py — Pydantic request and response models.

Single passenger OR batch (list of passengers) in one endpoint.
"""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator


class Passenger(BaseModel):
    """
    One Titanic passenger record.
    Only the columns available at prediction time — no Survived.
    """
    Pclass:   int   = Field(..., ge=1, le=3,  description="Passenger class (1, 2, or 3)")
    Sex:      str   = Field(...,               description="'male' or 'female'")
    Age:      float | None = Field(None, ge=0, le=120, description="Age in years (null = unknown)")
    SibSp:    int   = Field(0,  ge=0,          description="# of siblings / spouses aboard")
    Parch:    int   = Field(0,  ge=0,          description="# of parents / children aboard")
    Fare:     float | None = Field(None, ge=0, description="Passenger fare (null = unknown)")
    Embarked: str | None = Field(None,         description="Port: 'S', 'C', or 'Q' (null = unknown)")

    # Filler columns the preprocessor expects but aren't in test input
    Name:     str   = Field("Doe, Mr. John",   description="Passenger name (placeholder)")
    Ticket:   str   = Field("UNKNOWN",         description="Ticket number (placeholder)")
    Cabin:    str | None = Field(None,          description="Cabin number (optional)")
    PassengerId: int = Field(1,                 description="Passenger ID (auto-assigned)")

    @field_validator("Sex")
    @classmethod
    def validate_sex(cls, v: str) -> str:
        if v.lower() not in ("male", "female"):
            raise ValueError("Sex must be 'male' or 'female'")
        return v.lower()

    @field_validator("Embarked")
    @classmethod
    def validate_embarked(cls, v: str | None) -> str | None:
        if v is not None and v.upper() not in ("S", "C", "Q"):
            raise ValueError("Embarked must be 'S', 'C', or 'Q'")
        return v.upper() if v else None

    @field_validator("Pclass")
    @classmethod
    def validate_pclass(cls, v: int) -> int:
        if v not in (1, 2, 3):
            raise ValueError("Pclass must be 1, 2, or 3")
        return v


class PredictRequest(BaseModel):
    """Accepts one OR multiple passengers — same endpoint."""
    passengers: list[Passenger] = Field(
        ...,
        min_length=1,
        description="List of passenger records (1 = single, N = batch)",
    )


class PassengerPrediction(BaseModel):
    passenger_index: int
    survived:        int         # 0 or 1
    probability:     float       # probability of survival
    confidence:      str         # "High" / "Medium" / "Low"


class PredictResponse(BaseModel):
    predictions:      list[PassengerPrediction]
    total_passengers: int
    survived_count:   int
    model_name:       str
    model_source:     str        # "dagshub_registry" or "local_fallback"
    processing_time_ms: float


class HealthResponse(BaseModel):
    status:        str
    model_loaded:  bool
    model_name:    str
    model_source:  str
    model_version: str


class ModelInfoResponse(BaseModel):
    model_name:    str
    model_source:  str
    model_version: str
    model_type:    str
    features:      list[str]
    description:   str
