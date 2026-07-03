# SalesTeam AI — Intelligent Recommendation Agent

AI agent integrated into the SalesTeam Flutter app.
Automatically generates order proposals for sales reps
visiting client points of sale (LSAT subsidiary — ITech).

## How it works
1. Sales rep opens the client visit screen in Flutter
2. Flutter calls POST /api/recommend with the client_id
3. The AI agent returns a complete order proposal
4. The sales rep accepts / modifies / rejects each product
5. The final order + feedback is sent back to POST /api/feedback
6. Feedback is stored and used to retrain models weekly

## AI Parameters (6 fixed for MVP)
- Order history (frequency, recency, avg quantity, trend)
- Seasonality (month coefficient)
- Geographic location (for cold start only)
- Client type
- Product category and price
- New product flag (cold start)

## Installation

```bash
python -m venv venv
source venv/bin/activate      # Mac/Linux
venv\Scripts\activate         # Windows
pip install -r requirements.txt
cp .env.example .env          # fill in your API keys
```

## Run the API

```bash
uvicorn src.api.main:app --reload --port 8000
```

## API Endpoints
- POST /api/recommend  → get order proposal for a client
- POST /api/feedback   → record sales rep reaction
- GET  /health         → API status

## Data
- LSAT invoices    : ~18,444 invoices (Jan 2024 → Jun 2026)
- LSAT order lines : ~187,893 lines
- LSAT clients     : 737 points of sale (312 with GPS)
- Products         : 1,599 unique articles

## Stack
Python · FastAPI · XGBoost · Prophet · SVD · HuggingFace ·
pandas · scikit-learn · ngrok · MLflow
