import React, { useState, useEffect } from 'react';
import { 
  Sparkles, 
  User, 
  Sliders, 
  ShoppingBag, 
  CheckCircle2, 
  XCircle, 
  TrendingUp, 
  Search, 
  RefreshCw, 
  Info, 
  Award, 
  Zap,
  ArrowRight
} from 'lucide-react';

const API_BASE_URL = 'http://127.0.0.1:8000/api';

const PRESET_CLIENTS = [
  { id: 'CLT009160', label: 'CLT009160 (Téléphonie/Mobiles)' },
  { id: 'CLT010283', label: 'CLT010283 (Revendeur Pro)' },
  { id: 'CLT011029', label: 'CLT011029 (Point de Vente)' },
  { id: 'CLT011090', label: 'CLT011090 (Gros Volume)' },
  { id: 'CLT011292', label: 'CLT011292 (Boutique Tech)' },
  { id: 'CLT001977', label: 'CLT001977 (Client Régulier)' }
];

export default function App() {
  const [selectedClient, setSelectedClient] = useState('CLT009160');
  const [customClientInput, setCustomClientInput] = useState('');
  const [nbSuggestions, setNbSuggestions] = useState(8);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [recommendation, setRecommendation] = useState(null);
  const [itemStatuses, setItemStatuses] = useState({});
  const [quantities, setQuantities] = useState({});

  const fetchRecommendation = async (clientId) => {
    setLoading(true);
    setError(null);
    try {
      const response = await fetch(`${API_BASE_URL}/recommend`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          client_id: clientId,
          commercial_id: 'COMMERCIAL_LSAT',
          config: {
            use_order_history: true,
            use_seasonality: true,
            nb_suggestions: parseInt(nbSuggestions)
          }
        })
      });

      if (!response.ok) {
        throw new Error(`Erreur serveur (${response.status})`);
      }

      const data = await response.json();
      setRecommendation(data);
      
      // Initialize default quantities and statuses
      const initialQty = {};
      const initialStatus = {};
      data.suggestions.forEach((item) => {
        initialQty[item.code_article] = item.quantite_suggeree;
        initialStatus[item.code_article] = 'pending';
      });
      setQuantities(initialQty);
      setItemStatuses(initialStatus);
    } catch (err) {
      console.error(err);
      setError(err.message || "Impossible de contacter l'API de recommandation.");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchRecommendation(selectedClient);
  }, [selectedClient, nbSuggestions]);

  const handleSelectPreset = (clientId) => {
    setSelectedClient(clientId);
    setCustomClientInput('');
  };

  const handleCustomSubmit = (e) => {
    e.preventDefault();
    if (customClientInput.trim()) {
      setSelectedClient(customClientInput.trim());
    }
  };

  const toggleStatus = (codeArticle, newStatus) => {
    setItemStatuses(prev => ({
      ...prev,
      [codeArticle]: prev[codeArticle] === newStatus ? 'pending' : newStatus
    }));
  };

  const handleQtyChange = (codeArticle, val) => {
    const num = parseInt(val) || 1;
    setQuantities(prev => ({ ...prev, [codeArticle]: num }));
  };

  // Metrics calculation
  const totalSuggestions = recommendation?.suggestions?.length || 0;
  const highPriorityCount = recommendation?.suggestions?.filter(s => s.score_confiance >= 0.80).length || 0;
  const avgConfidence = totalSuggestions > 0 
    ? (recommendation.suggestions.reduce((acc, curr) => acc + curr.score_confiance, 0) / totalSuggestions * 100).toFixed(1)
    : 0;

  return (
    <div className="app-container">
      {/* App Header */}
      <header className="app-header">
        <div className="brand-title">
          <div className="brand-icon">
            <Sparkles size={24} />
          </div>
          <div>
            <h1>SalesTeam AI — Agent de Recommandation</h1>
            <p>Module de Test & Inférence en Temps Réel (XGBoost Classifier)</p>
          </div>
        </div>
        <div className="status-badge">
          <span className="status-dot"></span>
          FastAPI Backend Connecté (v1.0.0)
        </div>
      </header>

      {/* Main Grid */}
      <div className="dashboard-grid">
        {/* Left Sidebar Control Panel */}
        <aside>
          <div className="glass-card">
            <div className card-title>
              <User size={18} className="text-indigo" />
              Sélection du Client
            </div>

            {/* Quick Presets */}
            <div className="quick-preset-container">
              <div className="preset-label">Clients de Test Réels (LSAT)</div>
              <div className="preset-chips">
                {PRESET_CLIENTS.map((preset) => (
                  <button
                    key={preset.id}
                    className={`chip-button ${selectedClient === preset.id ? 'active' : ''}`}
                    onClick={() => handleSelectPreset(preset.id)}
                  >
                    {preset.id}
                  </button>
                ))}
              </div>
            </div>

            {/* Custom Client Search */}
            <form onSubmit={handleCustomSubmit} className="custom-input-group">
              <label>Rechercher par Code Client</label>
              <div className="input-with-button">
                <input
                  type="text"
                  placeholder="ex: CLT009160..."
                  className="input-field"
                  value={customClientInput}
                  onChange={(e) => setCustomClientInput(e.target.value)}
                />
                <button type="submit" className="btn-primary">
                  <Search size={16} />
                </button>
              </div>
            </form>

            {/* AI Config Sliders */}
            <div className="config-group">
              <div className card-title>
                <Sliders size={18} />
                Paramètres IA
              </div>
              
              <div className="config-item">
                <label>
                  <span>Nombre max de suggestions</span>
                  <span className="font-mono">{nbSuggestions}</span>
                </label>
                <input 
                  type="range" 
                  min="3" 
                  max="15" 
                  value={nbSuggestions} 
                  onChange={(e) => setNbSuggestions(e.target.value)}
                  className="range-slider"
                />
              </div>
            </div>
          </div>
        </aside>

        {/* Right Main Dashboard */}
        <main>
          {/* KPI Row */}
          <div className="kpi-row">
            <div className="kpi-card">
              <div className="kpi-icon-box kpi-icon-purple">
                <ShoppingBag size={22} />
              </div>
              <div>
                <div className="kpi-val">{totalSuggestions}</div>
                <div className="kpi-lbl">Articles Recommandés</div>
              </div>
            </div>

            <div className="kpi-card">
              <div className="kpi-icon-box kpi-icon-emerald">
                <Award size={22} />
              </div>
              <div>
                <div className="kpi-val">{highPriorityCount}</div>
                <div className="kpi-lbl">Priorité Haute (&gt;80%)</div>
              </div>
            </div>

            <div className="kpi-card">
              <div className="kpi-icon-box kpi-icon-cyan">
                <Zap size={22} />
              </div>
              <div>
                <div className="kpi-val">{avgConfidence}%</div>
                <div className="kpi-lbl">Confiance Moyenne IA</div>
              </div>
            </div>
          </div>

          {/* Proposal Card Table */}
          <div className="glass-card">
            <div className="card-title" style={{ justifyContent: 'space-between' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                <TrendingUp size={20} style={{ color: '#6366f1' }} />
                <span>Projet de Commande Généré pour <strong>{selectedClient}</strong></span>
              </div>
              <button 
                onClick={() => fetchRecommendation(selectedClient)} 
                className="btn-icon" 
                title="Actualiser la recommandation"
              >
                <RefreshCw size={16} />
              </button>
            </div>

            {loading && (
              <div className="empty-state">
                <div className="spinner"></div>
                <p>Calcul des probabilités par XGBoost en cours pour le client {selectedClient}...</p>
              </div>
            )}

            {error && (
              <div className="empty-state" style={{ color: '#f43f5e' }}>
                <XCircle size={40} style={{ margin: '0 auto 12px' }} />
                <p>{error}</p>
                <button onClick={() => fetchRecommendation(selectedClient)} className="btn-primary" style={{ marginTop: '16px' }}>
                  Réessayer
                </button>
              </div>
            )}

            {!loading && !error && recommendation?.suggestions?.length === 0 && (
              <div className="empty-state">
                <Info size={40} style={{ margin: '0 auto 12px', opacity: 0.5 }} />
                <p>Aucun produit n'a atteint le seuil d'achat recommandé pour ce client.</p>
              </div>
            )}

            {!loading && !error && recommendation?.suggestions?.length > 0 && (
              <div className="proposal-table-container">
                <table className="proposal-table">
                  <thead>
                    <tr>
                      <th>Article</th>
                      <th>Probabilité Achat</th>
                      <th>Statut IA</th>
                      <th>Qté Suggérée</th>
                      <th>Explication IA</th>
                      <th>Action Rep</th>
                    </tr>
                  </thead>
                  <tbody>
                    {recommendation.suggestions.map((item) => {
                      const probPct = (item.score_confiance * 100).toFixed(1);
                      const isHigh = item.score_confiance >= 0.80;
                      const status = itemStatuses[item.code_article] || 'pending';

                      return (
                        <tr key={item.code_article}>
                          <td>
                            <div className="article-title">{item.designation}</div>
                            <div style={{ display: 'flex', gap: '8px', alignItems: 'center', marginTop: '4px' }}>
                              <span className="article-code">{item.code_article}</span>
                              <span className="article-cat">• {item.categorie}</span>
                            </div>
                          </td>

                          <td>
                            <div className="proba-cell">
                              <div className="proba-bar-bg">
                                <div 
                                  className={`proba-bar-fill ${isHigh ? 'fill-high' : 'fill-mid'}`}
                                  style={{ width: `${probPct}%` }}
                                ></div>
                              </div>
                              <span className="proba-text" style={{ color: isHigh ? '#34d399' : '#fbbf24' }}>
                                {probPct}%
                              </span>
                            </div>
                          </td>

                          <td>
                            <span className={`badge ${isHigh ? 'badge-high' : 'badge-mid'}`}>
                              {isHigh ? 'Priorité Haute' : 'Recommandé'}
                            </span>
                          </td>

                          <td>
                            <input 
                              type="number"
                              min="1"
                              max="100"
                              value={quantities[item.code_article] || item.quantite_suggeree}
                              onChange={(e) => handleQtyChange(item.code_article, e.target.value)}
                              className="input-field font-mono"
                              style={{ width: '64px', padding: '6px 8px', textAlign: 'center' }}
                            />
                          </td>

                          <td>
                            <div className="explanation-text">{item.explication}</div>
                          </td>

                          <td>
                            <div className="action-buttons">
                              <button 
                                className={`btn-icon accept ${status === 'accepted' ? 'active-accept' : ''}`}
                                onClick={() => toggleStatus(item.code_article, 'accepted')}
                                title="Accepter la suggestion"
                              >
                                <CheckCircle2 size={16} />
                              </button>
                              <button 
                                className={`btn-icon reject ${status === 'rejected' ? 'active-reject' : ''}`}
                                onClick={() => toggleStatus(item.code_article, 'rejected')}
                                title="Rejeter la suggestion"
                              >
                                <XCircle size={16} />
                              </button>
                            </div>
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        </main>
      </div>
    </div>
  );
}
