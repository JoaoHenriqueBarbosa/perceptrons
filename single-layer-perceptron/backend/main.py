from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import sqlite3
from datetime import datetime
from typing import List
import json
import os
from multi_perceptron import MultiPerceptron, load_patterns_from_db

app = FastAPI()

# Configurar CORS para permitir requisições do frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Inicializar banco de dados
def init_db():
    conn = sqlite3.connect('patterns.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS training_patterns (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            pattern TEXT NOT NULL,
            label TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

init_db()

class Pattern(BaseModel):
    pattern: List[List[bool]]
    label: str = ""

@app.post("/api/patterns")
async def save_pattern(pattern: Pattern):
    # Converter a matriz para string para salvar no banco
    pattern_str = ','.join([''.join(['1' if cell else '0' for cell in row]) for row in pattern.pattern])

    conn = sqlite3.connect('patterns.db')
    cursor = conn.cursor()
    cursor.execute(
        'INSERT INTO training_patterns (pattern, label) VALUES (?, ?)',
        (pattern_str, pattern.label)
    )
    conn.commit()
    pattern_id = cursor.lastrowid
    conn.close()

    return {"id": pattern_id, "message": "Pattern saved successfully"}

@app.get("/api/patterns")
async def get_patterns():
    conn = sqlite3.connect('patterns.db')
    cursor = conn.cursor()
    cursor.execute('SELECT id, pattern, label, created_at FROM training_patterns ORDER BY created_at DESC')
    rows = cursor.fetchall()
    conn.close()

    patterns = []
    for row in rows:
        pattern_id, pattern_str, label, created_at = row
        # Reconstruir a matriz 16x16
        # Dividir por vírgula para separar as linhas, depois converter cada caractere
        pattern_rows = pattern_str.split(',')
        pattern_list = [[c == '1' for c in row] for row in pattern_rows]
        patterns.append({
            "id": pattern_id,
            "pattern": pattern_list,
            "label": label,
            "created_at": created_at
        })

    return patterns

@app.delete("/api/patterns/{pattern_id}")
async def delete_pattern(pattern_id: int):
    conn = sqlite3.connect('patterns.db')
    cursor = conn.cursor()
    cursor.execute('DELETE FROM training_patterns WHERE id = ?', (pattern_id,))
    deleted_rows = cursor.rowcount
    conn.commit()
    conn.close()

    if deleted_rows > 0:
        return {"message": "Pattern deleted successfully"}
    else:
        return {"message": "Pattern not found"}, 404

class TrainingConfig(BaseModel):
    epochs: int
    neurons: int

class PredictionRequest(BaseModel):
    model_id: str
    pattern: List[List[bool]]

class BulkLabelUpdate(BaseModel):
    old_label: str
    new_label: str

@app.post("/api/train")
async def train_model(config: TrainingConfig):
    """
    Treina um modelo com N neurônios.
    """
    try:
        # Carregar dados do banco
        X, y = load_patterns_from_db('patterns.db')

        # Verificar se há dados suficientes
        if len(X) == 0:
            raise HTTPException(status_code=400, detail="Nenhum padrão no banco de dados")

        unique_labels = len(set(y))
        max_classes = 2 ** config.neurons

        if unique_labels > max_classes:
            raise HTTPException(
                status_code=400,
                detail=f"Número de classes ({unique_labels}) excede capacidade de {config.neurons} neurônios ({max_classes} classes)"
            )

        # Criar e treinar modelo
        model = MultiPerceptron(input_size=256, num_neurons=config.neurons, learning_rate=0.01)
        train_info = model.train(X, y, epochs=config.epochs)

        # Avaliar
        metrics = model.evaluate(X, y)

        # Salvar modelo
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_filename = f"models/model_{config.neurons}n_{timestamp}.json"
        os.makedirs('models', exist_ok=True)
        model.save_model(model_filename)

        # Salvar info do modelo no banco
        conn = sqlite3.connect('patterns.db')
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS trained_models (
                id TEXT PRIMARY KEY,
                neurons INTEGER,
                epochs INTEGER,
                accuracy REAL,
                classes_count INTEGER,
                model_path TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        model_id = f"{config.neurons}n_{timestamp}"
        cursor.execute(
            '''INSERT INTO trained_models
               (id, neurons, epochs, accuracy, classes_count, model_path)
               VALUES (?, ?, ?, ?, ?, ?)''',
            (model_id, config.neurons, train_info['epochs_executed'],
             metrics['accuracy'], unique_labels, model_filename)
        )
        conn.commit()
        conn.close()

        return {
            "model_id": model_id,
            "neurons": config.neurons,
            "epochs_executed": train_info['epochs_executed'],
            "convergence": train_info['convergence'],
            "accuracy": metrics['accuracy'],
            "classes_count": unique_labels,
            "label_mapping": train_info['label_mapping']
        }

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro no treinamento: {str(e)}")

@app.get("/api/models")
async def get_models():
    """
    Lista todos os modelos treinados.
    """
    conn = sqlite3.connect('patterns.db')
    cursor = conn.cursor()

    # Criar tabela se não existir
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS trained_models (
            id TEXT PRIMARY KEY,
            neurons INTEGER,
            epochs INTEGER,
            accuracy REAL,
            classes_count INTEGER,
            model_path TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    cursor.execute('''
        SELECT id, neurons, epochs, accuracy, classes_count, created_at
        FROM trained_models
        ORDER BY created_at DESC
    ''')
    rows = cursor.fetchall()
    conn.close()

    models = []
    for row in rows:
        model_id, neurons, epochs, accuracy, classes_count, created_at = row
        models.append({
            "id": model_id,
            "neurons": neurons,
            "epochs": epochs,
            "accuracy": accuracy,
            "classes_count": classes_count,
            "created_at": created_at
        })

    return models

@app.delete("/api/models/{model_id}")
async def delete_model(model_id: str):
    """
    Deleta um modelo treinado.
    """
    try:
        conn = sqlite3.connect('patterns.db')
        cursor = conn.cursor()
        cursor.execute('DELETE FROM trained_models WHERE id = ?', (model_id,))
        deleted_rows = cursor.rowcount
        conn.commit()
        conn.close()
        if deleted_rows > 0:
            return {"message": "Modelo deletado com sucesso"}
        else:
            return {"message": "Modelo não encontrado"}, 404
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao deletar modelo: {str(e)}")

@app.put("/api/patterns/bulk-update-labels")
async def bulk_update_labels(update: BulkLabelUpdate):
    """
    Atualiza labels em massa.
    """
    try:
        conn = sqlite3.connect('patterns.db')
        cursor = conn.cursor()

        # Atualizar patterns com old_label para new_label
        cursor.execute(
            'UPDATE training_patterns SET label = ? WHERE label = ? OR (label IS NULL AND ? = "")',
            (update.new_label, update.old_label, update.old_label)
        )

        updated_count = cursor.rowcount
        conn.commit()
        conn.close()

        return {
            "message": f"{updated_count} padrões atualizados",
            "updated_count": updated_count,
            "old_label": update.old_label,
            "new_label": update.new_label
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao atualizar labels: {str(e)}")

@app.post("/api/predict")
async def predict_pattern(request: PredictionRequest):
    """
    Faz predição de um padrão usando modelo treinado.
    """
    try:
        # Buscar caminho do modelo no banco
        conn = sqlite3.connect('patterns.db')
        cursor = conn.cursor()
        cursor.execute('SELECT model_path, neurons FROM trained_models WHERE id = ?', (request.model_id,))
        row = cursor.fetchone()
        conn.close()

        if not row:
            raise HTTPException(status_code=404, detail="Modelo não encontrado")

        model_path, neurons = row

        # Verificar se arquivo existe
        if not os.path.exists(model_path):
            raise HTTPException(status_code=404, detail="Arquivo do modelo não encontrado")

        # Carregar modelo
        import numpy as np
        model = MultiPerceptron(input_size=256, num_neurons=neurons)
        model.load_model(model_path)

        # Converter padrão para numpy array
        pattern_matrix = [[int(cell) for cell in row] for row in request.pattern]
        pattern_flat = np.array(pattern_matrix).flatten()

        # Fazer predição
        prediction = model.predict(pattern_flat)

        # Obter código binário
        binary_code = tuple(neuron.predict(pattern_flat) for neuron in model.neurons)
        binary_str = ''.join(map(str, binary_code))

        return {
            "prediction": prediction,
            "binary_code": binary_str,
            "neurons": neurons
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro na predição: {str(e)}")

@app.get("/")
async def root():
    return {"message": "Perceptrons API"}
