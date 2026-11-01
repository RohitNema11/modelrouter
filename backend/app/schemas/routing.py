from enum import StrEnum

from pydantic import BaseModel, Field


class TaskType(StrEnum):
    GENERAL = "general"
    CODING = "coding"
    DEBUGGING = "debugging"
    REASONING = "reasoning"
    MATHEMATICS = "mathematics"
    SUMMARIZATION = "summarization"
    EXTRACTION = "extraction"
    CREATIVE = "creative"


class QueryAnalysis(BaseModel):
    """The local analysis of a query before model selection."""

    task_type: TaskType
    complexity: float = Field(ge=0.0, le=1.0)
    required_capabilities: list[str]
    reasoning: str


class RoutingDecision(BaseModel):
    """The result of selecting a model for a query."""

    selected_model: str
    confidence: float = Field(ge=0.0, le=1.0)
    analysis: QueryAnalysis
    reasoning: str
