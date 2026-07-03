"""
LAYER 4 — API
Pydantic models defining the exact JSON structure
exchanged between Flutter and the API.
"""

from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime


class AIConfig(BaseModel):
    """
    Configuration sent by the backoffice to control
    which AI parameters are active.
    Each parameter can be toggled on/off.
    """
    use_order_history:  bool       = True
    use_seasonality:    bool       = True
    use_localisation:   bool       = True
    use_client_type:    bool       = True
    use_category:       bool       = True
    use_promotions:     bool       = False   # phase 2
    filter_categories:  list[str]  = Field(default_factory=list)
    nb_suggestions:     int        = 5


class RecommendRequest(BaseModel):
    """Request sent by Flutter to get an order proposal."""
    client_id:     str            = Field(..., example="CLT011712")
    commercial_id: str            = Field(..., example="ML")
    visit_date:    Optional[str]  = Field(None, example="2025-06-15")
    config:        AIConfig       = Field(default_factory=AIConfig)


class ProductSuggestion(BaseModel):
    """One product suggestion in the order proposal."""
    code_article:       str
    designation:        str
    categorie:          str
    quantite_suggeree:  int
    quantite_min:       int
    quantite_max:       int
    score_confiance:    float
    is_nouveau_produit: bool
    explication:        str


class RecommendResponse(BaseModel):
    """Full response sent back to Flutter."""
    client_id:      str
    commercial_id:  str
    nb_suggestions: int
    suggestions:    list[ProductSuggestion]
    generated_at:   str


class FeedbackItem(BaseModel):
    """Feedback for one product in the order."""
    code_article:    str
    accepte:         bool
    quantite_finale: int
    modifie:         bool


class FeedbackRequest(BaseModel):
    """Full feedback after the sales rep validates the order."""
    client_id:     str
    commercial_id: str
    code_facture:  Optional[str] = None
    items:         list[FeedbackItem]
    submitted_at:  Optional[str] = None


class FeedbackResponse(BaseModel):
    """Confirmation that feedback was recorded."""
    status:   str = "ok"
    message:  str
    nb_items: int


class HealthResponse(BaseModel):
    """API health check response."""
    status:        str  = "healthy"
    version:       str  = "1.0.0"
    models_loaded: bool = False
    timestamp:     str
