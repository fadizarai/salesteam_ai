import React, { useState, useEffect } from 'react';
import {
  Search,
  RefreshCw,
  CheckCircle2,
  XCircle,
  Info,
  Mic,
  X,
  Loader2,
  Sparkles,
  ChevronDown,
  ChevronUp,
  Users,
} from 'lucide-react';

const API_BASE_URL = 'http://127.0.0.1:8000/api';

const COMPANIES = ['LSAT', 'NEWTECH', 'ONETEL'];



const DEFAULT_PRESET_CLIENTS = [
  { id: 'CLT091206', label: 'CLT091206' },
  { id: 'CLT011712', label: 'CLT011712' },
  { id: 'CLT070730', label: 'CLT070730' },
  { id: 'CLT100521', label: 'CLT100521' },
  { id: 'CLT009160', label: 'CLT009160' },
  { id: 'CLT001977', label: 'CLT001977' },
];

const AI_TOGGLES = [
  { key: 'use_order_history', label: 'Historique des commandes' },
  { key: 'use_seasonality', label: 'Saisonnalité' },
  { key: 'use_localisation', label: 'Localisation GPS' },
];

// ── Maps each of the 4 section titles to an accent colour ──
const SECTION_META = {
  'Pourquoi ce produit ?': { color: '#1a56e8', bg: 'rgba(26,86,232,0.06)', border: 'rgba(26,86,232,0.15)' },
  'Pourquoi cette quantité ?': { color: '#059669', bg: 'rgba(5,150,105,0.06)', border: 'rgba(5,150,105,0.18)' },
  'Pourquoi ce classement ?': { color: '#d97706', bg: 'rgba(217,119,6,0.06)', border: 'rgba(217,119,6,0.18)' },
  "Pourquoi ce niveau d'urgence ?": { color: '#dc2626', bg: 'rgba(220,38,38,0.06)', border: 'rgba(220,38,38,0.18)' },
};

/**
 * Parses "## Title\nBody" markdown sections and renders each as a styled block.
 * Handles both "Pourquoi cette quantité ?" (with accent) and ASCII variants.
 */
function DetailedSections({ text }) {
  if (!text) return null;

  // Split on ## headings (keep the title in each chunk)
  const rawSections = text.split(/\n(?=## )/).filter(Boolean);

  const sections = rawSections.map((chunk) => {
    const newlineIdx = chunk.indexOf('\n');
    if (newlineIdx === -1) return { title: chunk.replace(/^##\s*/, '').trim(), body: '' };
    const title = chunk.slice(0, newlineIdx).replace(/^##\s*/, '').trim();
    const body = chunk.slice(newlineIdx + 1).trim();
    return { title, body };
  });

  if (sections.length === 0) {
    // Fallback: render as plain pre-formatted text
    return (
      <div style={{ whiteSpace: 'pre-line', fontSize: '0.875rem', color: '#374151', lineHeight: 1.65 }}>
        {text}
      </div>
    );
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
      {sections.map(({ title, body }, i) => {
        // Match regardless of accent differences (é vs e)
        const metaKey = Object.keys(SECTION_META).find(
          (k) => k.toLowerCase().replace(/[éè]/g, 'e') === title.toLowerCase().replace(/[éè]/g, 'e')
        );
        const meta = SECTION_META[metaKey] ?? {
          color: '#6b7280', bg: 'rgba(107,114,128,0.06)', border: 'rgba(107,114,128,0.2)',
        };

        return (
          <div
            key={i}
            style={{
              borderRadius: 10,
              background: meta.bg,
              border: `1px solid ${meta.border}`,
              padding: '13px 16px',
            }}
          >
            <div style={{
              display: 'flex', alignItems: 'center', gap: 7,
              marginBottom: body ? 7 : 0,
            }}>
              <span style={{ fontWeight: 700, fontSize: '0.8rem', color: meta.color, letterSpacing: '0.02em' }}>
                {title}
              </span>
            </div>
            {body && (
              <p style={{
                margin: 0, fontSize: '0.84rem', color: '#374151',
                lineHeight: 1.65, whiteSpace: 'pre-line',
              }}>
                {body}
              </p>
            )}
          </div>
        );
      })}
    </div>
  );
}

export default function App() {
  const [company, setCompany] = useState('LSAT');
  const [selectedClient, setSelectedClient] = useState('');
  const [searchQuery, setSearchQuery] = useState('');
  const [visitDate, setVisitDate] = useState(new Date().toISOString().split('T')[0]);
  const [hasSearched, setHasSearched] = useState(false);
  const [loadingStep, setLoadingStep] = useState('');
  const [inputError, setInputError] = useState(null);

  // Keep AI config toggles in state to send to backend in background
  const [aiToggles] = useState({
    use_order_history: true,
    use_seasonality: true,
    use_localisation: true,
  });

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [quantities, setQuantities] = useState({});
  const [itemStatuses, setItemStatuses] = useState({});
  const [selectedItemForModal, setSelectedItemForModal] = useState(null);
  const [detailedExplanation, setDetailedExplanation] = useState(null);
  const [detailedLoading, setDetailedLoading] = useState(false);
  const [detailedError, setDetailedError] = useState(null);
  const [recommendation, setRecommendation] = useState(null);
  const [chatOpen, setChatOpen] = useState(false);
  const [availableClients, setAvailableClients] = useState([]);
  const [clientListOpen, setClientListOpen] = useState(false);
  const [clientListLoading, setClientListLoading] = useState(true);

  // ── Fetch detailed explanation when a card modal opens ──
  useEffect(() => {
    if (!selectedItemForModal) {
      // Reset when modal closes
      setDetailedExplanation(null);
      setDetailedLoading(false);
      setDetailedError(null);
      return;
    }
    let cancelled = false;
    setDetailedExplanation(null);
    setDetailedError(null);
    setDetailedLoading(true);
    fetch(`${API_BASE_URL}/explain-detailed`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        client_id: selectedClient,
        code_article: selectedItemForModal.code_article,
      }),
    })
      .then((res) => {
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        return res.json();
      })
      .then((data) => {
        if (!cancelled) setDetailedExplanation(data.explication_detaillee);
      })
      .catch(() => {
        if (!cancelled) setDetailedError(true);
      })
      .finally(() => {
        if (!cancelled) setDetailedLoading(false);
      });
    return () => { cancelled = true; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedItemForModal]);

  // ── Load Client List from API on Mount ──
  useEffect(() => {
    const loadAvailableClients = async () => {
      setClientListLoading(true);
      try {
        const res = await fetch(`${API_BASE_URL}/clients`);
        if (res.ok) {
          const data = await res.json();
          if (data.clients && data.clients.length > 0) {
            setAvailableClients(data.clients.map((c) => c.code_client));
          }
        }
      } catch (err) {
        console.warn('Could not load client list:', err);
      } finally {
        setClientListLoading(false);
      }
    };
    loadAvailableClients();
  }, []);

  // ── API Call ──
  const fetchRecommendation = async (clientId) => {
    const activeId = clientId ? clientId.trim().toUpperCase() : '';
    if (!activeId) {
      setInputError('Veuillez saisir un code client.');
      return;
    }

    setLoading(true);
    setLoadingStep('Recherche du client...');
    setError(null);
    setInputError(null);

    const stepTimer = setTimeout(() => {
      setLoadingStep('Génération des recommandations...');
    }, 450);

    try {
      const response = await fetch(`${API_BASE_URL}/recommend`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          client_id: activeId,
          commercial_id: `COMMERCIAL_${company}`,
          company: company,
          visit_date: visitDate || null,
          config: {
            use_order_history: aiToggles.use_order_history,
            use_seasonality: aiToggles.use_seasonality,
            use_localisation: aiToggles.use_localisation,
          },
        }),
      });

      if (!response.ok) {
        throw new Error('Client introuvable. Vérifiez le code client et réessayez.');
      }

      const data = await response.json();
      setRecommendation(data);

      const initialQty = {};
      const initialStatus = {};
      (data.suggestions || []).forEach((item) => {
        initialQty[item.code_article] = item.quantite_suggeree;
        initialStatus[item.code_article] = 'pending';
      });
      setQuantities(initialQty);
      setItemStatuses(initialStatus);
    } catch (err) {
      console.error(err);
      setError('Client introuvable. Vérifiez le code client et réessayez.');
      setRecommendation(null);
    } finally {
      clearTimeout(stepTimer);
      setLoading(false);
      setLoadingStep('');
    }
  };

  // ── Handlers ──
  const handleSearchSubmit = (e) => {
    e.preventDefault();
    const targetCode = searchQuery.trim().toUpperCase();
    if (!targetCode) {
      setInputError('Veuillez saisir un code client.');
      return;
    }
    setSelectedClient(targetCode);
    setHasSearched(true);
    fetchRecommendation(targetCode);
  };

  const toggleStatus = (codeArticle, newStatus) => {
    setItemStatuses((prev) => ({
      ...prev,
      [codeArticle]: prev[codeArticle] === newStatus ? 'pending' : newStatus,
    }));
  };

  const handleQtyChange = (codeArticle, val) => {
    const num = parseInt(val) || 1;
    setQuantities((prev) => ({ ...prev, [codeArticle]: num }));
  };

  // ── Filtered suggestions ──
  const filteredSuggestions = (recommendation?.suggestions || []).filter(
    (item) => item.urgency_group !== 'decouvrir'
  );

  const urgentItems = filteredSuggestions.filter((item) => item.urgency_group === 'urgent');
  const recommendedItems = filteredSuggestions.filter((item) => item.urgency_group === 'recommande');

  const renderCardList = (items, urgencyTitle, urgencyEmoji, sectionClass) => {
    if (items.length === 0) return null;
    return (
      <div className={`urgency-section ${sectionClass}`} style={{ marginBottom: '32px' }}>
        <h3 className="urgency-section-title" style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '16px', fontSize: '1.1rem', fontWeight: 700 }}>
          <span>{urgencyEmoji}</span> {urgencyTitle}
          <span style={{ fontSize: '0.8rem', fontWeight: 600, color: '#4b5563', background: '#e5e7eb', padding: '2px 8px', borderRadius: '999px', marginLeft: '6px' }}>
            {items.length}
          </span>
        </h3>
        <div className="results-grid">
          {items.map((item) => {
            const MAX_SCORE = 3.0;
            const finalScore = item.score_final ?? item.score_confiance;
            const barWidth = Math.min((finalScore / MAX_SCORE) * 100, 100).toFixed(1);
            const status = itemStatuses[item.code_article] || 'pending';

            const isIA = item.source_quantite === 'IA';
            const cardClass = isIA ? 'card-source-ia' : 'card-source-historique';

            return (
              <div
                className={`result-card ${cardClass}`}
                key={item.code_article}
                onClick={() => setSelectedItemForModal(item)}
                style={{ cursor: 'pointer', position: 'relative', overflow: 'hidden' }}
              >
                <div className="card-source-badge-container" style={{ marginBottom: '8px' }}>
                  {isIA ? (
                    <span className="source-badge badge-ia">🤖 Prédiction IA</span>
                  ) : (
                    <span className="source-badge badge-historique">📊 Moyenne historique</span>
                  )}
                </div>

                <div className="result-card-header" style={{ marginTop: '4px' }}>
                  <div>
                    <div className="result-article-name" style={{ fontWeight: 600, fontSize: '0.95rem' }}>
                      {item.designation}
                    </div>
                    <div className="result-article-code">
                      {item.code_article} • {item.categorie}
                    </div>
                  </div>
                  <button
                    className="btn-ai-analysis-trigger"
                    onClick={(e) => {
                      e.stopPropagation();
                      setSelectedItemForModal(item);
                    }}
                  >
                    <Sparkles size={14} />
                    <span>Justifier</span>
                  </button>
                </div>

                {isIA ? (
                  /* CARD TYPE 2 — Source : "IA" */
                  <div>
                    <div className="result-proba-row" style={{ marginTop: '12px', marginBottom: '8px' }}>
                      <div className="proba-bar-bg">
                        <div
                          className="proba-bar-fill proba-fill-blue"
                          style={{ width: `${barWidth}%` }}
                        ></div>
                      </div>
                      <span className="proba-text">Confiance : {(item.score_confiance * 100).toFixed(0)}%</span>
                    </div>
                    <div style={{ fontSize: '0.8rem', color: '#4b5563', marginBottom: '12px' }}>
                      Intervalle estimé : <strong>{item.quantite_min}</strong> - <strong>{item.quantite_max}</strong> u.
                    </div>
                  </div>
                ) : (
                  /* CARD TYPE 1 — Source : "historique" */
                  <div style={{ marginTop: '12px', marginBottom: '12px' }}>
                    <div style={{ fontSize: '0.8rem', color: '#b45309', background: '#fffbeb', border: '1px solid #fef3c7', padding: '8px 10px', borderRadius: '6px', marginBottom: '8px', lineHeight: '1.3' }}>
                      ⚠️ Basé sur l'historique car la variance d'achat est trop élevée pour l'IA.
                    </div>
                    <div style={{ fontSize: '0.8rem', color: '#4b5563' }}>
                      Bornes de commande : <strong>{item.quantite_min}</strong> à <strong>{item.quantite_max}</strong> u.
                    </div>
                  </div>
                )}

                {/* Actions */}
                <div
                  className="result-actions"
                  onClick={(e) => e.stopPropagation()}
                  style={{ marginTop: 'auto' }}
                >
                  <input
                    type="number"
                    min="1"
                    max="1000"
                    value={
                      quantities[item.code_article] || item.quantite_suggeree
                    }
                    onChange={(e) =>
                      handleQtyChange(item.code_article, e.target.value)
                    }
                    className="qty-input"
                  />
                  <button
                    className={`btn-accept ${status === 'accepted' ? 'active' : ''}`}
                    onClick={() =>
                      toggleStatus(item.code_article, 'accepted')
                    }
                  >
                    <CheckCircle2 size={14} />
                    Accepter
                  </button>
                  <button
                    className={`btn-reject ${status === 'rejected' ? 'active' : ''}`}
                    onClick={() =>
                      toggleStatus(item.code_article, 'rejected')
                    }
                  >
                    <XCircle size={14} />
                    Rejeter
                  </button>
                </div>
              </div>
            );
          })}
        </div>
      </div>
    );
  };

  return (
    <>
      {/* ─── NAVBAR ─── */}
      <nav className="navbar">
        <div className="navbar-brand">
          <div className="navbar-logo">
            <Sparkles size={20} />
          </div>
          <span className="navbar-title">SalesTeam AI</span>
        </div>
        <div className="company-pills">
          {COMPANIES.map((c) => (
            <button
              key={c}
              className={`company-pill ${company === c ? 'active' : ''}`}
              onClick={() => setCompany(c)}
            >
              {c}
            </button>
          ))}
        </div>
      </nav>

      {/* ─── MAIN ─── */}
      <div className="main-container">


        {/* ── SEARCH CLIENT CARD ── */}
        <div className="client-search-card">
          <h2 className="search-section-subtitle">Recherche du client</h2>

          <form onSubmit={handleSearchSubmit} className="search-form">
            <div className="search-fields-row">
              <div className="search-field-group">
                <label htmlFor="client-code-input" className="search-field-label">
                  Code client
                </label>
                <div className="search-input-box">
                  <Search size={18} className="search-input-icon" />
                  <input
                    id="client-code-input"
                    type="text"
                    className="client-code-input"
                    placeholder="ex: CLT091206..."
                    value={searchQuery}
                    onChange={(e) => {
                      setSearchQuery(e.target.value);
                      if (inputError) setInputError(null);
                    }}
                  />
                </div>
              </div>

              <div className="search-field-group">
                <label htmlFor="visit-date-input" className="search-field-label">
                  Date de visite
                </label>
                <div className="search-input-box">
                  <input
                    id="visit-date-input"
                    type="date"
                    className="client-date-input"
                    value={visitDate}
                    onChange={(e) => setVisitDate(e.target.value)}
                  />
                </div>
              </div>

              <div className="search-field-group button-group">
                <button type="submit" className="btn-search-primary" disabled={loading}>
                  {loading ? (
                    <>
                      <Loader2 size={18} className="spin-icon" />
                      <span>Génération...</span>
                    </>
                  ) : (
                    <>
                      <Sparkles size={18} />
                      <span>Générer l'offre</span>
                    </>
                  )}
                </button>
              </div>
            </div>

            {inputError && (
              <div className="input-error-badge">
                {inputError}
              </div>
            )}
          </form>

          {/* ── COLLAPSIBLE CLIENT LIST PANEL ── */}
          <div className="client-list-panel">
            <button
              type="button"
              className="client-list-toggle"
              onClick={() => setClientListOpen(!clientListOpen)}
            >
              <div className="client-list-toggle-left">
                <Users size={16} />
                <span>Codes clients disponibles</span>
                {availableClients.length > 0 && (
                  <span className="client-list-count">{availableClients.length}</span>
                )}
              </div>
              {clientListOpen ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
            </button>
            {clientListOpen && (
              <div className="client-list-body">
                {clientListLoading ? (
                  <div className="client-list-loading">
                    <Loader2 size={16} className="spin-icon" />
                    <span>Chargement...</span>
                  </div>
                ) : availableClients.length === 0 ? (
                  <div className="client-list-empty">Aucun client trouvé.</div>
                ) : (
                  <div className="client-list-chips">
                    {availableClients.map((code) => (
                      <button
                        key={code}
                        type="button"
                        className={`client-list-chip ${searchQuery.toUpperCase() === code ? 'active' : ''}`}
                        onClick={() => {
                          setSearchQuery(code);
                          if (inputError) setInputError(null);
                        }}
                      >
                        {code}
                      </button>
                    ))}
                  </div>
                )}
              </div>
            )}
          </div>
        </div>

        {/* ── DIVIDER ── */}
        <hr className="search-results-divider" />

        {/* ── RESULTS SECTION ── */}
        <section className="results-section">
          <div className="results-header">
            <div>
              <div className="section-title">
                {selectedClient ? (
                  <>Projet de Commande — <strong>{selectedClient}</strong></>
                ) : (
                  <>Résultats / Recommandations</>
                )}
              </div>
              {recommendation && hasSearched && (
                <div className="section-subtitle">
                  {filteredSuggestions.length} article(s) recommandé(s)
                </div>
              )}
            </div>
            {selectedClient && recommendation && hasSearched && (
              <button
                className="btn-refresh"
                onClick={() => fetchRecommendation(selectedClient)}
                title="Actualiser"
              >
                <RefreshCw size={16} />
              </button>
            )}
          </div>

          {/* 1. Initial State before search */}
          {!hasSearched && !loading && !error && (
            <div className="state-message initial-state">
              <Info size={40} style={{ margin: '0 auto 12px', opacity: 0.4 }} />
              <p>Saisissez un code client ci-dessus puis cliquez sur <strong>« Générer l'offre »</strong> pour lancer les recommandations IA.</p>
            </div>
          )}

          {/* 2. Loading State */}
          {loading && (
            <div className="state-message">
              <div className="spinner"></div>
              <p style={{ fontWeight: 600, color: 'var(--primary)', fontSize: '1rem' }}>
                {loadingStep || 'Recherche du client...'}
              </p>
              <p style={{ fontSize: '0.82rem', color: '#6b7280', marginTop: '6px' }}>
                Traitement du modèle de prédiction par l'IA...
              </p>
            </div>
          )}

          {/* 3. Error State */}
          {!loading && error && (
            <div className="state-message" style={{ color: '#dc2626' }}>
              <XCircle size={44} style={{ margin: '0 auto 12px' }} />
              <p style={{ fontWeight: 600, fontSize: '0.98rem', marginBottom: '8px' }}>{error}</p>
              {selectedClient && (
                <button
                  className="btn-retry"
                  onClick={() => fetchRecommendation(selectedClient)}
                >
                  Réessayer
                </button>
              )}
            </div>
          )}

          {/* 4. Empty Results State */}
          {hasSearched && !loading && !error && filteredSuggestions.length === 0 && (
            <div className="state-message">
              <Info size={40} style={{ margin: '0 auto 12px', opacity: 0.4 }} />
              <p>Aucun produit recommandé pour le client {selectedClient}.</p>
            </div>
          )}

          {/* 5. Results Grid */}
          {hasSearched && !loading && !error && filteredSuggestions.length > 0 && (
            <div className="urgency-sections-wrapper">
              {renderCardList(urgentItems, "URGENT — Réapprovisionnement en retard", "⚡", "urgency-urgent")}
              {renderCardList(recommendedItems, "RECOMMANDÉ — Forte probabilité d'achat", "✅", "urgency-recommande")}
            </div>
          )}
        </section>
      </div>

      {/* ─── PREMIUM LLM EXPLANATION MODAL ─── */}
      {selectedItemForModal && (
        <div
          className="modal-overlay"
          onClick={() => setSelectedItemForModal(null)}
        >
          <div
            className="modal-card"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="modal-header">
              <div className="modal-title-row">
                <div className="modal-icon-badge">
                  <Sparkles size={20} />
                </div>
                <div>
                  <h3 className="modal-title">Analyse & Justification IA</h3>
                  <p className="modal-subtitle">SalesTeam Intelligence • Llama 3.3</p>
                </div>
              </div>
              <button
                className="modal-close-btn"
                onClick={() => setSelectedItemForModal(null)}
              >
                <X size={20} />
              </button>
            </div>

            <div className="modal-body">
              <div className="modal-product-box">
                <div className="modal-product-name">
                  {selectedItemForModal.designation}
                </div>
                <div className="modal-product-code">
                  Code: <code>{selectedItemForModal.code_article}</code> • Catégorie: <strong>{selectedItemForModal.categorie}</strong>
                </div>
              </div>

              <div className="modal-stats-row">
                <div className="modal-stat-card">
                  <div className="stat-label">Score Final IA</div>
                  <div className="stat-value highlight">
                    {(selectedItemForModal.score_final ?? selectedItemForModal.score_confiance).toFixed(2)}
                  </div>
                  <div style={{ fontSize: '0.72rem', color: '#6b7280', marginTop: '2px' }}>ML × timing × tendance</div>
                </div>
                <div className="modal-stat-card">
                  <div className="stat-label">Prob. ML Brute</div>
                  <div className="stat-value">
                    {(selectedItemForModal.score_confiance * 100).toFixed(1)}%
                  </div>
                  <div style={{ fontSize: '0.72rem', color: '#6b7280', marginTop: '2px' }}>XGBoost classifieur</div>
                </div>
                <div className="modal-stat-card">
                  <div className="stat-label">Quantité Recommandée</div>
                  <div className="stat-value">
                    {selectedItemForModal.quantite_suggeree} <span style={{ fontSize: '0.85rem', color: '#6b7280' }}>unités</span>
                  </div>
                  <div style={{ fontSize: '0.72rem', color: '#6b7280', marginTop: '2px' }}>XGBoost régresseur</div>
                </div>
              </div>

              {/* ── Detailed explanation area ── */}
              <div className="modal-llm-quote-box" style={{ padding: 0, background: 'none', border: 'none' }}>
                {detailedLoading && (
                  <div style={{
                    display: 'flex', alignItems: 'center', gap: 10,
                    padding: '18px 20px', borderRadius: 12,
                    background: 'rgba(26,86,232,0.06)', border: '1px solid rgba(26,86,232,0.15)',
                  }}>
                    <Loader2 size={18} style={{ animation: 'spin 1s linear infinite', color: '#1a56e8', flexShrink: 0 }} />
                    <span style={{ fontSize: '0.875rem', color: '#6b7280' }}>Analyse en cours — génération de l'explication détaillée...</span>
                  </div>
                )}

                {detailedError && !detailedLoading && (
                  <div style={{
                    padding: '14px 18px', borderRadius: 10,
                    background: 'rgba(239,68,68,0.07)', border: '1px solid rgba(239,68,68,0.2)',
                    color: '#b91c1c', fontSize: '0.875rem',
                  }}>
                    Analyse détaillée indisponible pour le moment.
                  </div>
                )}

                {detailedExplanation && !detailedLoading && (
                  <DetailedSections text={detailedExplanation} />
                )}

                {!detailedLoading && !detailedError && !detailedExplanation && (
                  <div className="llm-quote-header" style={{ marginBottom: 0 }}>
                    <Sparkles size={16} color="#1a56e8" />
                    <span style={{ fontSize: '0.875rem', color: '#6b7280' }}>
                      &ldquo;{selectedItemForModal.explication}&rdquo;
                    </span>
                  </div>
                )}
              </div>
            </div>

            <div className="modal-footer">
              <button
                className="btn-primary"
                onClick={() => setSelectedItemForModal(null)}
                style={{ width: '100%', padding: '12px' }}
              >
                Fermer l'analyse
              </button>
            </div>
          </div>
        </div>
      )}

    </>
  );
}
