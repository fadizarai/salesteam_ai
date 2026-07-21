### 📄 `src/data/loader.py` — Reading the Excel Files

This file has loading functions for each raw source file. It reads them and renames their columns to standardize them. It does **not** perform any company filtering or cross-table joins anymore.


```python
# Reads the commande_phonesTech Excel file
# Loads all invoice rows without company filters
# Renames columns to French-friendly names:
#   No_              → code_facture   (invoice number)
#   client_code      → code_client    (client ID)
#   Posting Date     → date_commande  (order date)
#   Salesperson Code → ref_commercial (salesperson ID)

```


#### `load_lignes()` — Load order lines


```python
# Reads commande_lines_PhonesTech Excel
# Loads all order line rows without company filters.
# In the corrected dataset, LSAT is correctly labeled in the company column.
# Renames:
#   Document No_       → code_facture
#   Quantity           → quantite
#   designation_article → designation
#   Item Category Code  → categorie

```


#### `load_clients()` — Load GPS coordinates


```python
# Reads client-lat-lng Excel
# Renames:
#   clientCode → code_client
#   lat        → latitude
#   lng        → longitude
# Note: Only 312 out of 737 clients have GPS coordinates

```

#### `clean_all()` (in `cleaner.py`) — Clean and save separately


```python
# This is the MAIN function that:
# 1. Loads invoices (LSAT only)
# 2. Loads order lines (filtered by invoice codes → LSAT trick!)
# 3. Merges invoices + lines (JOIN on code_facture)
# 4. Merges with GPS (LEFT JOIN → clients without GPS get NaN)
# 5. Adds time columns: mois, annee, jour_semaine
# 6. Removes outliers: keeps only 0 < quantity <= 1000
# 7. Saves the result to data/processed/lsat_main_table.csv

```


**What a JOIN is** (for beginners):
```
Invoice table:          Order lines table:
code_facture | client   code_facture | product | qty
FAC001       | CLT01    FAC001       | IPHN15  | 5
FAC002       | CLT02    FAC001       | CASE01  | 10
                        FAC002       | CHGR01  | 3

After JOIN:
code_facture | client | product | qty
FAC001       | CLT01  | IPHN15  | 5
FAC001       | CLT01  | CASE01  | 10
FAC002       | CLT02  | CHGR01  | 3
```

---

### 📄 `src/data/cleaner.py` — Cleaning the Data

Three cleaning functions called in sequence by `clean()`:

#### `remove_duplicates()`
If the same product appears twice on the same invoice (data entry error), keep only one.

#### `handle_nulls()`


```python
# code_client or code_article is null → DELETE the row (can't work without these)
# categorie is null   → replace with "UNKNOWN"
# designation is null → replace with the product code itself
# latitude/longitude  → LEAVE as null (not all clients have GPS — that's normal)

```


#### `normalize_text_fields()`


```python
# "  iPhone " → "iPhone"  (strip whitespace)
# "smartphones" → "SMARTPHONES"  (uppercase categories)
# Prevents treating "PHONE" and "Phone" as two different categories

```


---

## 📁 Part 6 — `src/features/` — Layer 2: Building AI Inputs

**Key concept**: Machine Learning models don't understand raw data. They need **numbers**. This file transforms order history into a table of numbers — one row per (client, product) pair.

---

### 📄 `src/features/feature_engineering.py` — The 6 AI Parameters

The AI uses 6 types of information (called **features** or **parameters**) to make predictions.

#### Parameter 1 — Order History (`compute_order_history_features`)

For each (client, product) pair:

| Feature | What it is | Example |
|---------|-----------|---------|
| `avg_qty` | Average quantity ordered | 8.5 units |
| `std_qty` | How variable the quantity is | 2.3 (low = consistent buyer) |
| `frequency` | How many times ordered total | 12 times |
| `recency_days` | Days since last order | 45 days ago |
| `avg_delay_days` | Avg days between orders | 30 days = monthly buyer |
| `trend` | Is the client buying MORE or LESS recently? | +0.3 = growing demand |

**The trend calculation**:
```
Split the orders in 2 halves:
  Old orders: [5, 6, 5] → avg = 5.3
  New orders: [8, 9, 10] → avg = 9.0
  Trend = (9.0 - 5.3) / 5.3 = +0.66 → strong growth!
```

#### Parameter 2 — Seasonality (`compute_seasonality_features`)


```python
SEASONAL_COEF = {
    1:  0.85,   # January  → slow (post-holidays)
    3:  1.10,   # March    → Ramadan starts, spending increases
    4:  1.20,   # April    → Ramadan peak
    7:  1.25,   # July     → Summer peak (Tunisia specific)
    12: 1.30,   # December → Year-end highest demand
}

```


A coefficient of `1.30` for December means: multiply the normal quantity by 1.30 → suggest 30% more stock.

#### Parameter 3 — Geography (`compute_geo_features`)

Uses GPS coordinates to find **neighboring clients** within 5km:
```
Client A at (36.81, 10.17)
→ 8 other clients within 5km radius
→ If those 8 all buy product X, maybe Client A should too (cold start)
```
Uses a mathematical structure called **BallTree** for fast geographic search.

#### Parameters 4 + 5 + 6 — Product Info (`compute_product_features`)

| Feature | Meaning |
|---------|---------|
| `categorie` | e.g., "SMARTPHONE", "ACCESSORIES" |
| `is_new_product` | True if launched < 90 days ago |
| `days_on_market` | 15 days → brand new; 500 days → established |
| `total_clients` | How many clients have ever ordered it |

#### `build_feature_matrix()` — Joins all features

Creates the **master table** used by the AI:
```
One row = one (client, product) pair

client    | product  | avg_qty | frequency | recency | trend | month_coef | is_new
CLT011712 | IPHN15   | 8.5     | 12        | 45      | +0.3  | 1.20       | False
CLT011712 | CASE001  | 15.0    | 8         | 20      | 0.0   | 1.20       | False
CLT022301 | IPHN15   | 3.0     | 2         | 120     | -0.1  | 1.20       | False
...
```

---

## 📁 Part 7 — `src/models/` — Layer 2: The AI Models

---

### 📄 `src/models/train_classifier.py` — Should We Recommend This Product?

**This is the first AI question**: For client CLT011712 on a June visit, will they want to buy iPhone 15 cases?

- **Input**: The feature row for (CLT011712, CASE001)
- **Output**: `1` (yes, recommend it) or `0` (no, skip it)
- **Algorithm**: XGBoost — one of the most powerful classification algorithms today

**How it gets trained**:
```
Historical data:
  CLT011712 ordered CASE001 in June 2024 → label = 1 (positive example)
  CLT011712 did NOT order SAMSCASE in June 2024 → label = 0 (negative example)

The model learns: "What features predict a future order?"
```

---

### 📄 `src/models/train_regressor.py` — How Many Should We Suggest?

Once the classifier says "yes, recommend product X" → the regressor answers "suggest HOW MANY?"

- **Input**: Same feature row
- **Output**: A number (e.g., `8.3` → rounded to `8` units)
- **Algorithm**: XGBoost Regressor
- **Metrics**: MAE (on average, how far off is the prediction?), RMSE

---

### 📄 `src/models/train_svd.py` — Collaborative Filtering

**The Netflix idea**: "Clients who bought A also bought B."

SVD (Singular Value Decomposition) is the same math Netflix uses for recommendations.

```
Client × Product matrix:
              IPHN15  CASE001  CHGR001  SAMSNG15
CLT011712       5       10       0        3
CLT022301       8        0       5        7
CLT033401       3        8       2        0

SVD discovers hidden patterns:
→ "CLT011712 and CLT033401 have similar taste → if CLT033401 buys CHGR001, 
   maybe CLT011712 should too"
```

---

### 📄 `src/models/cold_start.py` — What If There's No History?

**The cold start problem**: What do you recommend for a brand-new product that nobody has bought yet? Or a new client with no purchase history?

**Solution 1 — New product** (content-based):
```
New product: "USB-C Cable Pro Max" (category: ACCESSORIES)
→ Find other ACCESSORIES that clients buy
→ Use those purchase patterns to estimate likely demand
```

**Solution 2 — New client** (geographic KNN):
```
New client CLT099 in Tunis suburb
→ Find 5 nearest existing clients by GPS
→ Aggregate their top products
→ "Your neighbors mostly buy chargers and phone cases → suggest those"
```

---

### 📄 `src/models/predictor.py` — The Brain's Control Center

This is the **orchestrator** of all AI models. It's a Python Class:


```python
class Predictor:
    # Loads ALL models into memory once when the API starts
    # Never loads from disk during a request (too slow!)
    
    def predict(client_id, config, visit_date):
        # Step 1: Does this client have history?
        #   YES → run classifier + regressor
        #   NO  → run cold_start_new_client()
        
        # Step 2: Are any products new?
        #   YES → run cold_start_new_product()
        
        # Step 3: Blend classifier + SVD scores
        #   final_score = 0.6 × classifier_score + 0.4 × svd_score
        
        # Step 4: Apply seasonality
        #   quantity × seasonal_coefficient
        
        # Step 5: Return top N suggestions sorted by confidence

```


---

## 📁 Part 8 — `src/services/` — Layer 3: Business Logic

---

### 📄 `src/services/recommendation.py` — The Full Pipeline

This service is called by the API route and orchestrates EVERYTHING:

```
Flutter sends: { client_id: "CLT011712", commercial_id: "ML" }
                    ↓
1. Load feature matrix for CLT011712
                    ↓
2. predictor.predict() → raw list of products + quantities + scores
                    ↓
3. For each product, call explanation.explain()
                    ↓
4. Package everything into a RecommendResponse
                    ↓
Flutter receives: { suggestions: [...], generated_at: "..." }
```

---

### 📄 `src/services/explanation.py` — The AI Writes French Explanations

For each suggestion, the AI generates a human-readable French explanation:

> *"Ce client commande ce produit tous les 30 jours en moyenne. La dernière commande date de 45 jours. La tendance est en hausse (+30%). Quantité suggérée : 10 unités."*

**How it works**:
1. **Call HuggingFace API** → send a prompt to Mistral-7B → get French text
2. **Cache for 24h** → don't call the API again for the same product/client/month
3. **Fallback templates** → if API is down, use pre-written French templates

**The cache** (in-memory dictionary):


```python
_explanation_cache = {
    "abc123": ("Ce client commande régulièrement...", expires_at=tomorrow),
    "def456": ("Nouveau produit lancé récemment...", expires_at=tomorrow),
}

```

Cache key = MD5 hash of (client_id + product_code + month)

---

### 📄 `src/services/feedback.py` — Learning from Sales Reps

After Ahmed visits a client and accepts/modifies/rejects suggestions:

```
FeedbackRequest:
{
  client_id: "CLT011712",
  items: [
    { code_article: "IPHN15", accepte: true, quantite_finale: 8, modifie: false },
    { code_article: "CASE001", accepte: true, quantite_finale: 20, modifie: true },
    { code_article: "CHGR001", accepte: false, quantite_finale: 0, modifie: false }
  ]
}
```

This gets saved to `data/feedback/feedback_2025-06.csv`. Each month has its own file.

**Weekly retraining**:
- Read all feedback CSVs
- Items accepted → positive training examples
- Items rejected → negative training examples
- Retrain classifier + regressor on this enriched dataset
- The AI gets smarter every week!

---

## 📁 Part 9 — `src/api/` — Layer 4: The HTTP Interface

This is how Flutter talks to the Python project.

---

### 📄 `src/api/schemas.py` — The Data Contracts

Defines exactly what JSON shapes are accepted and returned. Written with **Pydantic**, which automatically validates incoming data.


```python
# Flutter sends this:
RecommendRequest:
  client_id: "CLT011712"   ← required string
  commercial_id: "ML"      ← required string
  visit_date: "2025-06-15" ← optional
  config: { nb_suggestions: 5, use_seasonality: true, ... }

# Python returns this:
RecommendResponse:
  suggestions: [
    {
      code_article: "IPHN15",
      designation: "iPhone 15 Pro Case",
      quantite_suggeree: 8,
      quantite_min: 5,       ← minimum acceptable quantity
      quantite_max: 12,      ← maximum acceptable quantity
      score_confiance: 0.87, ← 87% confidence
      is_nouveau_produit: false,
      explication: "Ce client commande ce produit tous les 30 jours..."
    },
    ...
  ]

```


If Flutter sends wrong data (missing `client_id`), Pydantic automatically rejects it with a clear error message.

---

### 📄 `src/api/main.py` — The FastAPI App


```python
app = FastAPI(title="SalesTeam AI")

# CORS: allows Flutter (running on a different port/device) to call this API
app.add_middleware(CORSMiddleware, allow_origins=["*"])

# Register the 3 route groups:
app.include_router(recommend.router, prefix="/api")  # POST /api/recommend
app.include_router(feedback.router,  prefix="/api")  # POST /api/feedback
app.include_router(admin.router,     prefix="/api")  # GET /health, POST /retrain

# lifespan: runs code at startup (load models) and shutdown (cleanup)

```


When you run `uvicorn src.api.main:app --reload --port 8000`, this starts an HTTP server on port 8000 that Flutter can talk to.

---

### 📄 `src/api/routes/recommend.py` — POST /api/recommend

```
Flutter → POST http://your-ip:8000/api/recommend
          Body: { "client_id": "CLT011712", "commercial_id": "ML" }

Python  → validates request with Pydantic
        → calls recommendation.recommend(request)
        → returns the suggestion list

If error → returns HTTP 501 (not implemented) or 500 (server error)
```

---

### 📄 `src/api/routes/feedback.py` — POST /api/feedback

Same pattern. Flutter sends the salesperson's reactions, Python saves them to CSV.

---

### 📄 `src/api/routes/admin.py` — Admin Tools

- `GET /health` → Check the API is alive. Returns `{ "status": "healthy" }`
- `POST /retrain` → Trigger model retraining (to be implemented)

---

## 🔄 Part 10 — How All Files Connect (The Full Flow)

```
[Excel files in data/raw/]
         ↓ loader.py reads them
[Raw DataFrame: 78,720 order lines]
         ↓ cleaner.py cleans
[Clean DataFrame: ~78,530 rows]
         ↓ feature_engineering.py transforms
[Feature Matrix: 40,765 rows × 29 features]
         ↓ train_classifier.py learns from
[Trained XGBoost Classifier → models/classifier_lsat.pkl]
         ↓ train_regressor.py learns from
[Trained XGBoost Regressor  → models/regressor_lsat.pkl]
         ↓ train_svd.py learns from
[Trained SVD Model          → models/svd_lsat.pkl]

═══════════════════ API START ═══════════════════

uvicorn starts → main.py loads → Predictor loads all 3 .pkl files

Flutter → POST /api/recommend
              ↓ routes/recommend.py
              ↓ services/recommendation.py
              ↓ models/predictor.py (classifier + regressor + SVD + cold start)
              ↓ services/explanation.py (HuggingFace → French text)
              ↓ RecommendResponse (JSON)
Flutter ← { suggestions: [...] }

Sales rep reacts → POST /api/feedback
              ↓ services/feedback.py
              ↓ data/feedback/feedback_2025-06.csv (saved)

[Weekly] → models retrained on feedback → AI gets smarter
```

---

## 🧩 Part 11 — Key Concepts Glossary

| Term | Simple Explanation |
|------|--------------------|
| **DataFrame** | A table in Python (like an Excel sheet). Has rows and columns. From the `pandas` library. |
| **Feature** | A number that describes something. "45 days since last order" is a feature. |
| **Training** | Showing the AI many examples so it learns patterns. |
| **Classifier** | AI that answers YES/NO questions. |
| **Regressor** | AI that predicts a number (quantity). |
| **SVD** | Math that finds hidden patterns in a table. Used for "users who liked X also liked Y". |
| **Cold Start** | The problem of making recommendations when you have no history. |
| **API** | A door that other programs can knock on to get data. Flutter knocks, Python answers. |
| **Endpoint** | One specific door. `/api/recommend` is an endpoint. |
| **Pydantic** | A library that validates data shapes. Like a bouncer checking IDs at the door. |
| **JSON** | A text format for exchanging data between programs. `{ "key": "value" }` |
| **Cache** | Storing a result so you don't compute it again. "I already translated this, here's the saved translation." |
| **Feedback loop** | AI makes predictions → human corrects → AI learns from corrections → AI improves. |
| **pkl file** | A saved Python object. The trained AI model lives in a `.pkl` file on disk. |
| **HuggingFace** | A website with thousands of free AI models. Like GitHub but for AI. |
| **Mistral-7B** | A powerful French-speaking AI model (7 billion parameters). Used for explanations. |
