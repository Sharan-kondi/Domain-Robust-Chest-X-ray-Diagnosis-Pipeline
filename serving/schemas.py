"""
schemas.py — Pydantic schemas for the FastAPI inference API.
"""

from pydantic import BaseModel, Field
from typing import Dict, List, Optional


class PredictionResult(BaseModel):
    """Single pathology prediction."""
    label: str
    probability: float = Field(ge=0.0, le=1.0)
    positive: bool


class PredictionResponse(BaseModel):
    """Full prediction response for one X-ray image."""
    predictions: List[PredictionResult]
    uncertainty: float = Field(description="Predictive entropy (higher = less certain)")
    needs_human_review: bool = Field(description="True if uncertainty exceeds threshold")
    confidence_threshold: float
    gradcam_available: bool = False
    gradcam_url: Optional[str] = None
    gradcam_image: Optional[str] = None  # Base64 encoded png string
    model_version: str = "1.0.0"


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = "healthy"
    model_loaded: bool
    device: str
    version: str = "1.0.0"


class DriftReport(BaseModel):
    """Data drift monitoring report."""
    drift_detected: bool
    drift_score: float
    n_samples_analyzed: int
    details: Optional[Dict] = None


class DiscrepancyItem(BaseModel):
    """Represents a single clinical discrepancy claim."""
    claim: Optional[str] = None
    finding: Optional[str] = None
    explanation: str


class GroundingCitation(BaseModel):
    """Citations mapping a pathology back to standard guidelines."""
    acr_code: str
    section: str
    recommendation: str
    citations: List[str]
    pathology: str


class VerificationResult(BaseModel):
    """Result of claims matching and spatial verification."""
    hallucinations: List[DiscrepancyItem]
    omissions: List[DiscrepancyItem]
    discrepancy_count: int
    error: Optional[str] = None


class ReportResponse(BaseModel):
    """Unified radiology AI prediction and agent generated report response."""
    prediction: PredictionResponse
    report: str
    verification: VerificationResult
    grounding: List[GroundingCitation]
    escalated: bool
    trace_id: str
    steps: List[Dict] = []
    bias: Optional[Dict] = None

