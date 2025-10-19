import { useState, useRef, useEffect } from 'react'

interface TrainedModel {
  id: string
  neurons: number
  epochs: number
  accuracy: number
  classes_count: number
}

interface PredictionResult {
  prediction: string
  confidence?: number
  binary_code?: string
}

export function Test() {
  const [grid, setGrid] = useState<boolean[][]>(() =>
    Array(16).fill(null).map(() => Array(16).fill(false))
  )
  const [isDrawing, setIsDrawing] = useState(false)
  const [models, setModels] = useState<TrainedModel[]>([])
  const [selectedModel, setSelectedModel] = useState<string>('')
  const [predictionResult, setPredictionResult] = useState<PredictionResult | null>(null)
  const [isTesting, setIsTesting] = useState(false)
  const drawMode = useRef<boolean>(true)

  const loadModels = async () => {
    try {
      const response = await fetch('http://localhost:8000/api/models')
      if (response.ok) {
        const data = await response.json()
        setModels(data)
        if (data.length > 0 && !selectedModel) {
          setSelectedModel(data[0].id)
        }
      }
    } catch (error) {
      console.error('Erro ao carregar modelos:', error)
    }
  }

  useEffect(() => {
    loadModels()
  }, [])

  const toggleCell = (row: number, col: number) => {
    setGrid(prev => {
      const newGrid = prev.map(r => [...r])
      newGrid[row][col] = drawMode.current
      return newGrid
    })
  }

  const handleMouseDown = (row: number, col: number) => {
    setIsDrawing(true)
    drawMode.current = true
    toggleCell(row, col)
  }

  const handleMouseEnter = (row: number, col: number) => {
    if (isDrawing) {
      setGrid(prev => {
        const newGrid = prev.map(r => [...r])
        newGrid[row][col] = drawMode.current
        return newGrid
      })
    }
  }

  const handleMouseUp = () => {
    setIsDrawing(false)
  }

  const clearGrid = () => {
    setGrid(Array(16).fill(null).map(() => Array(16).fill(false)))
    setPredictionResult(null)
  }

  const testPattern = async () => {
    if (!selectedModel) {
      alert('Selecione um modelo primeiro!')
      return
    }

    setIsTesting(true)
    setPredictionResult(null)

    try {
      const response = await fetch('http://localhost:8000/api/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          model_id: selectedModel,
          pattern: grid
        }),
      })

      if (response.ok) {
        const result = await response.json()
        setPredictionResult(result)
      } else {
        const error = await response.json()
        alert(`Erro: ${error.detail || 'Erro ao fazer predição'}`)
      }
    } catch (error) {
      alert('Erro ao conectar com a API')
      console.error(error)
    } finally {
      setIsTesting(false)
    }
  }

  return (
    <div className="page test-page" onMouseUp={handleMouseUp}>
      <div className="test-container">
        <div className="test-draw-section">
          <h2>Desenhe o Padrão</h2>

          <div className="test-controls">
            <div className="model-selector">
              <label>Modelo:</label>
              <select
                value={selectedModel}
                onChange={(e) => setSelectedModel(e.target.value)}
                disabled={models.length === 0}
              >
                {models.length === 0 && (
                  <option>Nenhum modelo disponível</option>
                )}
                {models.map((model) => (
                  <option key={model.id} value={model.id}>
                    {model.neurons} neurônio{model.neurons > 1 ? 's' : ''} -
                    {(model.accuracy * 100).toFixed(1)}%
                  </option>
                ))}
              </select>
            </div>

            <div className="test-buttons">
              <button onClick={clearGrid}>Limpar</button>
              <button
                className="test-btn"
                onClick={testPattern}
                disabled={isTesting || !selectedModel}
              >
                {isTesting ? 'Testando...' : 'Testar Padrão'}
              </button>
            </div>
          </div>

          <div className="grid">
            {grid.map((row, rowIndex) => (
              <div key={rowIndex} className="grid-row">
                {row.map((cell, colIndex) => (
                  <div
                    key={`${rowIndex}-${colIndex}`}
                    className={`cell ${cell ? 'active' : ''}`}
                    onMouseDown={() => handleMouseDown(rowIndex, colIndex)}
                    onMouseEnter={() => handleMouseEnter(rowIndex, colIndex)}
                  />
                ))}
              </div>
            ))}
          </div>
        </div>

        <div className="test-result-section">
          <h2>Resultado</h2>

          {models.length === 0 && (
            <div className="no-models-message">
              <p>Nenhum modelo treinado ainda!</p>
              <p className="hint">Vá para a página Training para treinar um modelo primeiro.</p>
            </div>
          )}

          {predictionResult && (
            <div className="prediction-result">
              <div className="prediction-label">
                <span className="label-text">Predição:</span>
                <span className="prediction-value">{predictionResult.prediction}</span>
              </div>

              {predictionResult.binary_code && (
                <div className="prediction-details">
                  <div className="detail-item">
                    <span>Código binário:</span>
                    <code>{predictionResult.binary_code}</code>
                  </div>
                </div>
              )}

              <div className="result-explanation">
                O modelo classificou este padrão como: <strong>{predictionResult.prediction}</strong>
              </div>
            </div>
          )}

          {!predictionResult && selectedModel && (
            <div className="awaiting-test">
              <p>Desenhe um padrão e clique em "Testar Padrão"</p>
            </div>
          )}
        </div>
      </div>

      <div className="test-info">
        <h3>💡 Dica</h3>
        <p>
          Desenhe padrões semelhantes aos que você treinou para ver se o modelo consegue reconhecê-los!
        </p>
        <p className="warning">
          <strong>Importante:</strong> Para ter um modelo funcional, você precisa treinar com exemplos
          de DIFERENTES classes (ex: T, não-T, L, H, etc.)
        </p>
      </div>
    </div>
  )
}
