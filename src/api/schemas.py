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
    use_promotions:     bool       = False   # phase 2
    filter_categories:  list[str]  = Field(default_factory=list)


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
    source_quantite:    str     = "IA"  # "IA", "historique", "historique (variance trop élevée)"
    score_confiance:    float   # raw ML probability
    score_final:        float   # after business re-ranking (ML × timing × trend)
    timing_boost:       float   # 1.0 / 1.5 / 2.0 / 3.0 depending on reorder cycle
    is_nouveau_produit: bool
    explication:        str
    urgency_group:      str     # "urgent", "recommande", "decouvrir"
    recency_relative:   float


class RecommendResponse(BaseModel):
    """Full response sent back to Flutter."""
    client_id:      str
    commercial_id:  str
    # TODO: nb_suggestions actuellement non lu par recommendation.py —
    # les plafonds urgent(7)/recommande(5) restent fixes. Décision en attente
    # sur un usage futur (ex: limite d'affichage frontend indépendante du
    # split urgent/recommandé).
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
