import { useState, useEffect } from 'react'

interface TrainingConfig {
  epochs: number
  hidden_layers: number[]
  learning_rate: number
  batch_size: number
}

interface TrainedModel {
  id: string
  architecture: number[]
  epochs: number
  accuracy: number
  final_loss: number
  learning_rate: number
  created_at: string
}

export function Training() {
  const [config, setConfig] = useState<TrainingConfig>({
    epochs: 300,
    hidden_layers: [64, 32],
    learning_rate: 0.3,
    batch_size: 16
  })
  const [isTraining, setIsTraining] = useState(false)
  const [trainedModels, setTrainedModels] = useState<TrainedModel[]>([])
  const [trainingLog, setTrainingLog] = useState<string[]>([])
  const [hiddenLayersInput, setHiddenLayersInput] = useState('64,32')

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
    setTrainingLog(['Iniciando treinamento...'])

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
          `✓ Treinamento concluído!`,
          ``,
          `Arquitetura: ${result.architecture.join(' → ')}`,
          `Épocas: ${result.epochs}`,
          `Loss final: ${result.final_loss.toFixed(4)}`,
          `Acurácia final: ${(result.final_accuracy * 100).toFixed(2)}%`,
          ``,
          `Classes detectadas: ${result.classes_count}`,
          `Mapeamento: ${JSON.stringify(result.label_mapping, null, 2)}`
        ])
        loadModels()
      } else {
        const error = await response.json()
        setTrainingLog([`⚠ Erro: ${error.detail || 'Erro ao treinar'}`])
      }
    } catch (error) {
      setTrainingLog([`⚠ Erro de conexão: ${error}`])
    } finally {
      setIsTraining(false)
    }
  }

  const deleteModel = async (modelId: string) => {
    try {
      const response = await fetch(`http://localhost:8000/api/models/${modelId}`, {
        method: 'DELETE',
      })
      if (response.ok) {
        loadModels()
      } else {
        console.error('Erro ao deletar modelo')
      }
    } catch (error) {
      console.error('Erro ao deletar modelo:', error)
    }
  }

  const handleHiddenLayersChange = (value: string) => {
    setHiddenLayersInput(value)
    try {
      // Parse input like "128,64" or "128, 64" into array
      const layers = value
        .split(',')
        .map(s => parseInt(s.trim()))
        .filter(n => !isNaN(n) && n > 0)

      if (layers.length > 0) {
        setConfig({ ...config, hidden_layers: layers })
      }
    } catch (error) {
      console.error('Erro ao parsear camadas:', error)
    }
  }

  return (
    <div className="page training-page">
      <div className="training-container">
        <div className="training-config">
          <h2>Configuração de Treinamento - MLP</h2>

          <div className="config-section">
            <label>
              Camadas Ocultas
              <div className="info-text">
                Arquitetura: 256 → {config.hidden_layers.join(' → ')} → (auto)
              </div>
            </label>
            <input
              type="text"
              placeholder="Ex: 128,64 ou 128"
              value={hiddenLayersInput}
              onChange={(e) => handleHiddenLayersChange(e.target.value)}
              disabled={isTraining}
              className="text-input"
            />
            <div className="info-text small">
              Digite os tamanhos das camadas ocultas separados por vírgula.
              Ex: "128,64" cria uma rede 256 → 128 → 64 → saída
            </div>
          </div>

          <div className="config-section">
            <label>
              Épocas de Treinamento
              <div className="info-text">{config.epochs} iterações</div>
            </label>
            <input
              type="range"
              min="50"
              max="1000"
              step="50"
              value={config.epochs}
              onChange={(e) => setConfig({ ...config, epochs: parseInt(e.target.value) })}
              disabled={isTraining}
            />
            <div className="range-values">
              <span>50</span>
              <span>1000</span>
            </div>
          </div>

          <div className="config-section">
            <label>
              Learning Rate
              <div className="info-text">{config.learning_rate.toFixed(2)}</div>
            </label>
            <input
              type="range"
              min="0.05"
              max="0.5"
              step="0.05"
              value={config.learning_rate}
              onChange={(e) => setConfig({ ...config, learning_rate: parseFloat(e.target.value) })}
              disabled={isTraining}
            />
            <div className="range-values">
              <span>0.05</span>
              <span>0.50</span>
            </div>
          </div>

          <div className="config-section">
            <label>
              Batch Size
              <div className="info-text">{config.batch_size} amostras por batch</div>
            </label>
            <input
              type="range"
              min="8"
              max="64"
              step="8"
              value={config.batch_size}
              onChange={(e) => setConfig({ ...config, batch_size: parseInt(e.target.value) })}
              disabled={isTraining}
            />
            <div className="range-values">
              <span>8</span>
              <span>64</span>
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
              <pre className="log-content">
                {trainingLog.join('\n')}
              </pre>
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
                    <h3>MLP</h3>
                    <span className="model-accuracy">
                      {(model.accuracy * 100).toFixed(1)}%
                    </span>
                  </div>
                  <div className="model-details">
                    <div className="architecture-text">
                      {model.architecture.join(' → ')}
                    </div>
                    <div>Épocas: {model.epochs}</div>
                    <div>Loss: {model.final_loss.toFixed(4)}</div>
                    <div>LR: {model.learning_rate}</div>
                    <div className="model-date">
                      {new Date(model.created_at).toLocaleString('pt-BR')}
                    </div>
                  </div>
                  <button
                    className="delete-btn"
                    onClick={() => deleteModel(model.id)}
                    disabled={isTraining}
                  >
                    Deletar
                  </button>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>

      <div className="training-info">
        <h3>Multi-Layer Perceptron (MLP)</h3>
        <ul>
          <li><strong>Forward Propagation:</strong> Dados fluem da entrada através das camadas ocultas até a saída</li>
          <li><strong>Função de Ativação:</strong> Sigmoid em todas as camadas (σ(z) = 1/(1+e^(-z)))</li>
          <li><strong>Backpropagation:</strong> Algoritmo que calcula gradientes para ajustar pesos</li>
          <li><strong>Mini-batch:</strong> Divide dados em lotes para treinamento mais eficiente</li>
          <li><strong>Cross-Entropy Loss:</strong> Função de custo para classificação multi-classe</li>
          <li><strong>Gradient Clipping:</strong> Previne explosão de gradientes</li>
          <li><strong>Early Stopping:</strong> Para o treinamento se não houver melhoria</li>
        </ul>

        <h4>💡 Configurações Recomendadas:</h4>
        <ul>
          <li><strong>Arquitetura:</strong> 64,32 ou 128,64 (não exagere!)</li>
          <li><strong>Learning Rate:</strong> 0.2-0.3 (mais alto = converge rápido)</li>
          <li><strong>Batch Size:</strong> 16-32 (menor = mais atualizações)</li>
          <li><strong>Épocas:</strong> 200-400 (early stopping evita overfitting)</li>
        </ul>

        <p className="info-note">
          ⚠️ <strong>Evite:</strong> Arquiteturas muito grandes (ex: 512,256,128) causam overfitting em datasets pequenos!
        </p>
      </div>
    </div>
  )
}
