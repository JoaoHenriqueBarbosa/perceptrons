import { useState, useEffect } from 'react'

interface TrainingConfig {
  epochs: number
  neurons: number
}

interface TrainedModel {
  id: string
  neurons: number
  epochs: number
  accuracy: number
  created_at: string
}

export function Training() {
  const [config, setConfig] = useState<TrainingConfig>({
    epochs: 100,
    neurons: 1
  })
  const [isTraining, setIsTraining] = useState(false)
  const [trainedModels, setTrainedModels] = useState<TrainedModel[]>([])
  const [trainingLog, setTrainingLog] = useState<string[]>([])

  const loadModels = async () => {
    try {
      const response = await fetch('http://localhost:8000/api/models')
      if (response.ok) {
        const data = await response.json()
        setTrainedModels(data)
      }
    } catch (error) {
      console.error('Erro ao carregar modelos:', error)
    }
  }

  useEffect(() => {
    loadModels()
  }, [])

  const startTraining = async () => {
    setIsTraining(true)
    setTrainingLog([])

    try {
      const response = await fetch('http://localhost:8000/api/train', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(config),
      })

      if (response.ok) {
        const result = await response.json()
        setTrainingLog([
          `Treinamento concluído!`,
          `Neurônios: ${result.neurons}`,
          `Épocas executadas: ${result.epochs_executed}`,
          `Acurácia: ${(result.accuracy * 100).toFixed(2)}%`,
          `Classes detectadas: ${result.classes_count}`,
          result.convergence ? `✓ Convergiu!` : `⚠ Não convergiu`
        ])
        loadModels()
      } else {
        const error = await response.json()
        setTrainingLog([`Erro: ${error.detail || 'Erro ao treinar'}`])
      }
    } catch (error) {
      setTrainingLog([`Erro de conexão: ${error}`])
    } finally {
      setIsTraining(false)
    }
  }

  const getMaxClasses = (neurons: number) => {
    return Math.pow(2, neurons)
  }

  return (
    <div className="page training-page">
      <div className="training-container">
        <div className="training-config">
          <h2>Configuração de Treinamento</h2>

          <div className="config-section">
            <label>
              Número de Neurônios
              <div className="info-text">
                {config.neurons} neurônio{config.neurons > 1 ? 's' : ''} = até {getMaxClasses(config.neurons)} classes
              </div>
            </label>
            <input
              type="range"
              min="1"
              max="4"
              value={config.neurons}
              onChange={(e) => setConfig({ ...config, neurons: parseInt(e.target.value) })}
              disabled={isTraining}
            />
            <div className="range-labels">
              <span>1 neurônio<br/>(2 classes)</span>
              <span>2 neurônios<br/>(4 classes)</span>
              <span>3 neurônios<br/>(8 classes)</span>
              <span>4 neurônios<br/>(16 classes)</span>
            </div>
          </div>

          <div className="config-section">
            <label>
              Épocas de Treinamento
              <div className="info-text">{config.epochs} iterações</div>
            </label>
            <input
              type="range"
              min="10"
              max="500"
              step="10"
              value={config.epochs}
              onChange={(e) => setConfig({ ...config, epochs: parseInt(e.target.value) })}
              disabled={isTraining}
            />
            <div className="range-values">
              <span>10</span>
              <span>500</span>
            </div>
          </div>

          <button
            className="train-btn"
            onClick={startTraining}
            disabled={isTraining}
          >
            {isTraining ? 'Treinando...' : 'Iniciar Treinamento'}
          </button>

          {trainingLog.length > 0 && (
            <div className="training-log">
              <h3>Resultado</h3>
              {trainingLog.map((line, i) => (
                <div key={i} className="log-line">{line}</div>
              ))}
            </div>
          )}
        </div>

        <div className="models-section">
          <h2>Modelos Treinados ({trainedModels.length})</h2>

          {trainedModels.length === 0 ? (
            <p className="empty-message">Nenhum modelo treinado ainda</p>
          ) : (
            <div className="models-grid">
              {trainedModels.map((model) => (
                <div key={model.id} className="model-card">
                  <div className="model-header">
                    <h3>{model.neurons} Neurônio{model.neurons > 1 ? 's' : ''}</h3>
                    <span className="model-accuracy">
                      {(model.accuracy * 100).toFixed(1)}%
                    </span>
                  </div>
                  <div className="model-details">
                    <div>Épocas: {model.epochs}</div>
                    <div>Max classes: {getMaxClasses(model.neurons)}</div>
                    <div className="model-date">
                      {new Date(model.created_at).toLocaleString('pt-BR')}
                    </div>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>

      <div className="training-info">
        <h3>Como funciona?</h3>
        <ul>
          <li><strong>1 neurônio:</strong> Classificação binária (ex: T vs não-T)</li>
          <li><strong>2 neurônios:</strong> Até 4 classes usando codificação binária (00, 01, 10, 11)</li>
          <li><strong>3 neurônios:</strong> Até 8 classes (000, 001, 010, 011, 100, 101, 110, 111)</li>
          <li><strong>N neurônios:</strong> Até 2^N classes</li>
        </ul>
        <p className="info-note">
          Cada neurônio aprende uma "pergunta binária". A combinação das respostas identifica a classe.
        </p>
      </div>
    </div>
  )
}
