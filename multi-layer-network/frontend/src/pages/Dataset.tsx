import { useState, useRef, useEffect } from 'react'

interface SavedPattern {
  id: number
  pattern: boolean[][]
  label: string
  created_at: string
}

export function Dataset() {
  const [grid, setGrid] = useState<boolean[][]>(() =>
    Array(16).fill(null).map(() => Array(16).fill(false))
  )
  const [isDrawing, setIsDrawing] = useState(false)
  const [savedPatterns, setSavedPatterns] = useState<SavedPattern[]>([])
  const [selectedPattern, setSelectedPattern] = useState<SavedPattern | null>(null)
  const [label, setLabel] = useState<string>('')
  const drawMode = useRef<boolean>(true)

  const toggleCell = (row: number, col: number) => {
    setGrid(prev => {
      const newGrid = prev.map(r => [...r])
      newGrid[row][col] = drawMode.current
      return newGrid
    })
  }

  const handleMouseDown = (row: number, col: number) => {
    setIsDrawing(true)
    drawMode.current = true // Sempre pintar, nunca apagar
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
  }

  const loadPatterns = async () => {
    try {
      const response = await fetch('http://localhost:8000/api/patterns')
      if (response.ok) {
        const data = await response.json()
        setSavedPatterns(data)
      }
    } catch (error) {
      console.error('Erro ao carregar patterns:', error)
    }
  }

  useEffect(() => {
    loadPatterns()
  }, [])

  const savePattern = async () => {
    try {
      const response = await fetch('http://localhost:8000/api/patterns', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          pattern: grid,
          label: label
        }),
      })

      if (response.ok) {
        loadPatterns() // Recarregar lista
        clearGrid() // Limpar grid para novo desenho
        setLabel('') // Limpar label também
      } else {
        console.error('Erro ao salvar padrão')
      }
    } catch (error) {
      console.error('Erro ao conectar com a API:', error)
    }
  }

  const deletePattern = async (id: number) => {
    try {
      const response = await fetch(`http://localhost:8000/api/patterns/${id}`, {
        method: 'DELETE',
      })

      if (response.ok) {
        // Se o pattern deletado era o selecionado, limpar seleção
        if (selectedPattern?.id === id) {
          setSelectedPattern(null)
        }
        loadPatterns() // Recarregar lista
      } else {
        console.error('Erro ao deletar padrão')
      }
    } catch (error) {
      console.error('Erro ao conectar com a API:', error)
    }
  }

  const updateAllLabels = async () => {
    const newLabel = prompt('Atualizar TODOS os padrões sem label para qual classe?', 'T')
    if (!newLabel) return

    try {
      const response = await fetch('http://localhost:8000/api/patterns/bulk-update-labels', {
        method: 'PUT',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          old_label: '',
          new_label: newLabel
        }),
      })

      if (response.ok) {
        const data = await response.json()
        alert(`${data.updated_count} padrões atualizados para "${newLabel}"`)
        loadPatterns()
      } else {
        console.error('Erro ao atualizar labels')
      }
    } catch (error) {
      console.error('Erro ao conectar com a API:', error)
    }
  }

  return (
    <div className="page" onMouseUp={handleMouseUp}>
      <div className="main-content">
        <div className="draw-section">
          <h2>Desenhar Padrão</h2>
          <div className="label-input">
            <label htmlFor="pattern-label">Label (classe):</label>
            <input
              id="pattern-label"
              type="text"
              value={label}
              onChange={(e) => setLabel(e.target.value)}
              placeholder="ex: T, não-T, L, etc."
            />
          </div>
          <div className="controls">
            <button onClick={clearGrid}>Limpar</button>
            <button onClick={savePattern}>Salvar Exemplo</button>
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

        <div className="patterns-section">
          <div className="patterns-header">
            <h2>Padrões Salvos ({savedPatterns.length})</h2>
            <button className="bulk-update-btn" onClick={updateAllLabels}>
              Atualizar Labels sem classe
            </button>
          </div>

          {savedPatterns.length > 0 && (
            <div className="dataset-stats">
              <h3>Estatísticas por Classe:</h3>
              <div className="stats-grid">
                {Object.entries(
                  savedPatterns.reduce((acc, p) => {
                    const label = p.label || 'sem label'
                    acc[label] = (acc[label] || 0) + 1
                    return acc
                  }, {} as Record<string, number>)
                ).map(([label, count]) => (
                  <div key={label} className="stat-item">
                    <span className={`label-badge ${label === 'sem label' ? 'no-label' : ''}`}>
                      {label}
                    </span>
                    <span className="stat-count">{count} exemplos</span>
                  </div>
                ))}
              </div>
            </div>
          )}
          <div className="patterns-container">
            <div className="patterns-table">
              <table>
                <thead>
                  <tr>
                    <th>ID</th>
                    <th>Label</th>
                    <th>Data</th>
                    <th>Ações</th>
                  </tr>
                </thead>
                <tbody>
                  {savedPatterns.map((pattern) => (
                    <tr
                      key={pattern.id}
                      className={selectedPattern?.id === pattern.id ? 'selected' : ''}
                    >
                      <td>{pattern.id}</td>
                      <td>
                        <span className={`label-badge ${pattern.label ? '' : 'no-label'}`}>
                          {pattern.label || 'sem label'}
                        </span>
                      </td>
                      <td>{new Date(pattern.created_at).toLocaleString('pt-BR')}</td>
                      <td>
                        <div className="action-buttons">
                          <button onClick={() => setSelectedPattern(pattern)}>
                            Ver
                          </button>
                          <button
                            className="delete-btn"
                            onClick={() => deletePattern(pattern.id)}
                          >
                            Apagar
                          </button>
                        </div>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
              {savedPatterns.length === 0 && (
                <p className="empty-message">Nenhum padrão salvo ainda</p>
              )}
            </div>

            {selectedPattern && (
              <div className="pattern-preview">
                <h3>Padrão #{selectedPattern.id}</h3>
                <div className="grid preview">
                  {selectedPattern.pattern.map((row, rowIndex) => (
                    <div key={rowIndex} className="grid-row">
                      {row.map((cell, colIndex) => (
                        <div
                          key={`${rowIndex}-${colIndex}`}
                          className={`cell ${cell ? 'active' : ''}`}
                        />
                      ))}
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}
