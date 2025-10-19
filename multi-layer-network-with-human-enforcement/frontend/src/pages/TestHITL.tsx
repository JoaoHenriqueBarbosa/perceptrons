import { useState, useEffect } from 'react'

interface Model {
  id: string
  architecture: number[]
  accuracy: number
}

interface PredictionResult {
  prediction: string
  uncertainty: number
  needs_feedback: boolean
  probabilities: Record<string, number>
}

interface FeedbackStats {
  model_id: string
  model_stats: {
    total_feedback: number
    improvements: number
    improvement_rate: number
  }
  db_feedback_count: number
  feedback_by_label: Record<string, number>
}

export function TestHITL() {
  const [grid, setGrid] = useState<boolean[][]>(
    Array(16).fill(null).map(() => Array(16).fill(false))
  )
  const [isDrawing, setIsDrawing] = useState(false)
  const [models, setModels] = useState<Model[]>([])
  const [selectedModel, setSelectedModel] = useState<string>('')
  const [prediction, setPrediction] = useState<PredictionResult | null>(null)
  const [showFeedbackForm, setShowFeedbackForm] = useState(false)
  const [feedbackLabel, setFeedbackLabel] = useState('')
  const [feedbackMessage, setFeedbackMessage] = useState('')
  const [stats, setStats] = useState<FeedbackStats | null>(null)

  useEffect(() => {
    loadModels()
  }, [])

  const loadModels = async () => {
    try {
      const response = await fetch('http://localhost:8000/api/models')
      if (response.ok) {
        const data = await response.json()
        setModels(data)
        if (data.length > 0) {
          setSelectedModel(data[0].id)
          loadStats(data[0].id)
        }
      }
    } catch (error) {
      console.error('Erro ao carregar modelos:', error)
    }
  }

  const loadStats = async (modelId: string) => {
    try {
      const response = await fetch(`http://localhost:8000/api/feedback-stats/${modelId}`)
      if (response.ok) {
        const data = await response.json()
        setStats(data)
      }
    } catch (error) {
      console.error('Erro ao carregar estatísticas:', error)
    }
  }

  const handleMouseDown = (row: number, col: number) => {
    setIsDrawing(true)
    toggleCell(row, col)
  }

  const handleMouseEnter = (row: number, col: number) => {
    if (isDrawing) {
      toggleCell(row, col)
    }
  }

  const handleMouseUp = () => {
    setIsDrawing(false)
  }

  const toggleCell = (row: number, col: number) => {
    const newGrid = grid.map((r, i) =>
      r.map((cell, j) => (i === row && j === col ? !cell : cell))
    )
    setGrid(newGrid)
  }

  const clearGrid = () => {
    setGrid(Array(16).fill(null).map(() => Array(16).fill(false)))
    setPrediction(null)
    setShowFeedbackForm(false)
    setFeedbackMessage('')
  }

  const testPattern = async () => {
    if (!selectedModel) {
      alert('Selecione um modelo primeiro!')
      return
    }

    try {
      const response = await fetch('http://localhost:8000/api/predict-with-uncertainty', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          model_id: selectedModel,
          pattern: grid,
        }),
      })

      if (response.ok) {
        const result = await response.json()
        setPrediction(result)
        setShowFeedbackForm(false)
        setFeedbackMessage('')
      } else {
        const error = await response.json()
        alert(`Erro: ${error.detail}`)
      }
    } catch (error) {
      alert(`Erro de conexão: ${error}`)
    }
  }

  const submitFeedback = async () => {
    if (!feedbackLabel || !prediction) return

    try {
      const response = await fetch('http://localhost:8000/api/submit-feedback', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          model_id: selectedModel,
          pattern: grid,
          predicted_label: prediction.prediction,
          correct_label: feedbackLabel,
          uncertainty: prediction.uncertainty,
        }),
      })

      if (response.ok) {
        const result = await response.json()
        setFeedbackMessage(
          `✓ Feedback aplicado! ${result.improved ? 'Modelo melhorou!' : 'Modelo atualizado.'} (${result.total_feedback} feedbacks totais)`
        )
        setShowFeedbackForm(false)
        setFeedbackLabel('')

        // Recarregar estatísticas
        loadStats(selectedModel)
      } else {
        const error = await response.json()
        setFeedbackMessage(`⚠ Erro: ${error.detail}`)
      }
    } catch (error) {
      setFeedbackMessage(`⚠ Erro de conexão: ${error}`)
    }
  }

  const getUncertaintyColor = (uncertainty: number) => {
    if (uncertainty < 0.2) return '#4caf50' // Verde - confiante
    if (uncertainty < 0.4) return '#ff9800' // Laranja - médio
    return '#f44336' // Vermelho - incerto
  }

  return (
    <div className="page test-hitl-page">
      <div className="test-container">
        <div className="draw-section">
          <h2>Desenhe um padrão</h2>

          <div
            className="grid"
            onMouseLeave={handleMouseUp}
            style={{ userSelect: 'none' }}
          >
            {grid.map((row, i) => (
              <div key={i} className="grid-row">
                {row.map((cell, j) => (
                  <div
                    key={`${i}-${j}`}
                    className={`cell ${cell ? 'active' : ''}`}
                    onMouseDown={() => handleMouseDown(i, j)}
                    onMouseEnter={() => handleMouseEnter(i, j)}
                    onMouseUp={handleMouseUp}
                  />
                ))}
              </div>
            ))}
          </div>

          <div className="controls">
            <button onClick={clearGrid} className="clear-btn">
              Limpar
            </button>
            <button onClick={testPattern} className="test-btn" disabled={!selectedModel}>
              Testar Padrão
            </button>
          </div>

          <div className="model-selector">
            <label>Modelo:</label>
            <select
              value={selectedModel}
              onChange={(e) => {
                setSelectedModel(e.target.value)
                loadStats(e.target.value)
              }}
            >
              {models.map((model) => (
                <option key={model.id} value={model.id}>
                  {model.id} - {(model.accuracy * 100).toFixed(1)}%
                </option>
              ))}
            </select>
          </div>
        </div>

        <div className="results-section">
          <h2>Resultados com HITL</h2>

          {prediction && (
            <div className="prediction-result">
              <div className="prediction-header">
                <h3>Predição: <span className="prediction-label">{prediction.prediction}</span></h3>
                <div
                  className="uncertainty-badge"
                  style={{ backgroundColor: getUncertaintyColor(prediction.uncertainty) }}
                >
                  Incerteza: {(prediction.uncertainty * 100).toFixed(1)}%
                </div>
              </div>

              {prediction.needs_feedback && (
                <div className="active-learning-alert">
                  🤔 <strong>Active Learning:</strong> O modelo está incerto! Seu feedback ajudaria muito.
                </div>
              )}

              <div className="probabilities">
                <h4>Probabilidades:</h4>
                {Object.entries(prediction.probabilities).map(([label, prob]) => (
                  <div key={label} className="prob-bar">
                    <span className="prob-label">{label}</span>
                    <div className="prob-track">
                      <div
                        className="prob-fill"
                        style={{ width: `${prob * 100}%` }}
                      />
                    </div>
                    <span className="prob-value">{(prob * 100).toFixed(1)}%</span>
                  </div>
                ))}
              </div>

              {!showFeedbackForm && (
                <button
                  className="feedback-btn"
                  onClick={() => setShowFeedbackForm(true)}
                >
                  {prediction.prediction === 'T' || prediction.prediction === 'No-T'
                    ? '✏️ Corrigir Predição (RLHF)'
                    : '✏️ Fornecer Feedback'}
                </button>
              )}

              {showFeedbackForm && (
                <div className="feedback-form">
                  <h4>Qual é o label correto?</h4>
                  <div className="feedback-options">
                    <button
                      className={`feedback-option ${feedbackLabel === 'T' ? 'selected' : ''}`}
                      onClick={() => setFeedbackLabel('T')}
                    >
                      T
                    </button>
                    <button
                      className={`feedback-option ${feedbackLabel === 'No-T' ? 'selected' : ''}`}
                      onClick={() => setFeedbackLabel('No-T')}
                    >
                      No-T
                    </button>
                  </div>
                  <div className="feedback-actions">
                    <button onClick={submitFeedback} disabled={!feedbackLabel} className="submit-feedback-btn">
                      Enviar Feedback
                    </button>
                    <button onClick={() => setShowFeedbackForm(false)} className="cancel-btn">
                      Cancelar
                    </button>
                  </div>
                </div>
              )}

              {feedbackMessage && (
                <div className={`feedback-message ${feedbackMessage.includes('✓') ? 'success' : 'error'}`}>
                  {feedbackMessage}
                </div>
              )}
            </div>
          )}

          {!prediction && (
            <p className="empty-message">Desenhe um padrão e clique em "Testar Padrão"</p>
          )}
        </div>
      </div>

      {stats && (
        <div className="stats-section">
          <h3>📊 Estatísticas de Feedback</h3>
          <div className="stats-grid">
            <div className="stat-card">
              <div className="stat-value">{stats.model_stats.total_feedback}</div>
              <div className="stat-label">Total de Feedbacks</div>
            </div>
            <div className="stat-card">
              <div className="stat-value">{stats.model_stats.improvements}</div>
              <div className="stat-label">Melhorias</div>
            </div>
            <div className="stat-card">
              <div className="stat-value">
                {(stats.model_stats.improvement_rate * 100).toFixed(1)}%
              </div>
              <div className="stat-label">Taxa de Melhoria</div>
            </div>
          </div>

          {Object.keys(stats.feedback_by_label).length > 0 && (
            <div className="feedback-by-label">
              <h4>Feedbacks por Label:</h4>
              {Object.entries(stats.feedback_by_label).map(([label, count]) => (
                <div key={label} className="label-count">
                  <span>{label}:</span> <strong>{count}</strong>
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      <div className="hitl-info">
        <h3>🎯 Human-in-the-Loop Learning</h3>
        <ul>
          <li><strong>Active Learning:</strong> Sistema pede feedback quando incerto (entropia &gt; 30%)</li>
          <li><strong>RLHF:</strong> Seu feedback ajusta os pesos da rede imediatamente</li>
          <li><strong>Aprendizado Contínuo:</strong> Modelo melhora com cada correção</li>
        </ul>
        <div className="uncertainty-legend">
          <h4>Legenda de Incerteza:</h4>
          <div className="legend-item">
            <div className="legend-color" style={{ backgroundColor: '#4caf50' }} />
            <span>&lt; 20% - Confiante</span>
          </div>
          <div className="legend-item">
            <div className="legend-color" style={{ backgroundColor: '#ff9800' }} />
            <span>20-40% - Médio</span>
          </div>
          <div className="legend-item">
            <div className="legend-color" style={{ backgroundColor: '#f44336' }} />
            <span>&gt; 40% - Incerto (pede feedback)</span>
          </div>
        </div>
      </div>
    </div>
  )
}
