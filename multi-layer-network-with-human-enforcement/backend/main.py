from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import sqlite3
from datetime import datetime
from typing import List
import json
import os
from mlp import MLP, load_patterns_from_db
from mlp_hitl import MLP_HITL
import numpy as np

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

    # Tabela para armazenar feedback humano
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS human_feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model_id TEXT NOT NULL,
            pattern TEXT NOT NULL,
            predicted_label TEXT,
            correct_label TEXT,
            uncertainty REAL,
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
    hidden_layers: List[int]  # Ex: [128, 64] para duas camadas ocultas
    learning_rate: float = 0.1
    batch_size: int = 32

class PredictionRequest(BaseModel):
    model_id: str
    pattern: List[List[bool]]

class BulkLabelUpdate(BaseModel):
    old_label: str
    new_label: str

@app.post("/api/train")
async def train_model(config: TrainingConfig):
    """
    Treina um modelo MLP (Multi-Layer Perceptron).
    """
    try:
        # Carregar dados do banco
        X, y = load_patterns_from_db('patterns.db')

        # Verificar se há dados suficientes
        if len(X) == 0:
            raise HTTPException(status_code=400, detail="Nenhum padrão no banco de dados")

        num_classes = len(set(y))

        # Construir arquitetura: entrada -> camadas ocultas -> saída
        layer_sizes = [256] + config.hidden_layers + [num_classes]

        # Criar e treinar modelo
        model = MLP(layer_sizes=layer_sizes, learning_rate=config.learning_rate)
        train_info = model.train(X, y, epochs=config.epochs, batch_size=config.batch_size)

        # Avaliar
        metrics = model.evaluate(X, y)

        # Salvar modelo
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        arch_str = '_'.join(map(str, layer_sizes))
        model_filename = f"models/mlp_{arch_str}_{timestamp}.json"
        os.makedirs('models', exist_ok=True)
        model.save_model(model_filename)

        # Salvar info do modelo no banco
        conn = sqlite3.connect('patterns.db')
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS trained_models (
                id TEXT PRIMARY KEY,
                architecture TEXT,
                epochs INTEGER,
                accuracy REAL,
                final_loss REAL,
                classes_count INTEGER,
                model_path TEXT,
                learning_rate REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        model_id = f"mlp_{timestamp}"
        cursor.execute(
            '''INSERT INTO trained_models
               (id, architecture, epochs, accuracy, final_loss, classes_count, model_path, learning_rate)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)''',
            (model_id, json.dumps(layer_sizes), train_info['epochs'],
             metrics['accuracy'], train_info['final_loss'], num_classes,
             model_filename, config.learning_rate)
        )
        conn.commit()
        conn.close()

        return {
            "model_id": model_id,
            "architecture": layer_sizes,
            "epochs": train_info['epochs'],
            "final_loss": train_info['final_loss'],
            "final_accuracy": train_info['final_accuracy'],
            "accuracy": metrics['accuracy'],
            "classes_count": num_classes,
            "label_mapping": train_info['label_mapping'],
            "history": train_info['history']
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
            architecture TEXT,
            epochs INTEGER,
            accuracy REAL,
            final_loss REAL,
            classes_count INTEGER,
            model_path TEXT,
            learning_rate REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    cursor.execute('''
        SELECT id, architecture, epochs, accuracy, final_loss, classes_count, learning_rate, created_at
        FROM trained_models
        ORDER BY created_at DESC
    ''')
    rows = cursor.fetchall()
    conn.close()

    models = []
    for row in rows:
        model_id, architecture, epochs, accuracy, final_loss, classes_count, learning_rate, created_at = row
        models.append({
            "id": model_id,
            "architecture": json.loads(architecture) if architecture else [],
            "epochs": epochs,
            "accuracy": accuracy,
            "final_loss": final_loss,
            "classes_count": classes_count,
            "learning_rate": learning_rate,
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
    Faz predição de um padrão usando modelo MLP treinado.
    """
    try:
        # Buscar caminho do modelo no banco
        conn = sqlite3.connect('patterns.db')
        cursor = conn.cursor()
        cursor.execute('SELECT model_path, architecture FROM trained_models WHERE id = ?', (request.model_id,))
        row = cursor.fetchone()
        conn.close()

        if not row:
            raise HTTPException(status_code=404, detail="Modelo não encontrado")

        model_path, architecture_json = row
        architecture = json.loads(architecture_json)

        # Verificar se arquivo existe
        if not os.path.exists(model_path):
            raise HTTPException(status_code=404, detail="Arquivo do modelo não encontrado")

        # Carregar modelo
        model = MLP(layer_sizes=architecture)
        model.load_model(model_path)

        # Converter padrão para numpy array
        pattern_matrix = [[int(cell) for cell in row] for row in request.pattern]
        pattern_flat = np.array(pattern_matrix).flatten()

        # Fazer predição
        prediction = model.predict(pattern_flat)

        # Obter probabilidades de todas as classes
        probabilities = model.predict_proba(pattern_flat)

        # Converter probabilidades para formato JSON-friendly
        probs_dict = {}
        for label, idx in model.label_to_index.items():
            probs_dict[label] = float(probabilities[0][idx])

        return {
            "prediction": prediction,
            "probabilities": probs_dict,
            "architecture": architecture
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro na predição: {str(e)}")

@app.get("/")
async def root():
    return {"message": "Multi-Layer Perceptron (MLP) with Human-in-the-Loop API"}


# ============================================================================
# HUMAN-IN-THE-LOOP ENDPOINTS
# ============================================================================

class PredictionWithUncertaintyRequest(BaseModel):
    model_id: str
    pattern: List[List[bool]]

class FeedbackRequest(BaseModel):
    model_id: str
    pattern: List[List[bool]]
    predicted_label: str
    correct_label: str
    uncertainty: float = 0.0

@app.post("/api/predict-with-uncertainty")
async def predict_with_uncertainty(request: PredictionWithUncertaintyRequest):
    """
    Faz predição com cálculo de incerteza (Active Learning).
    """
    try:
        # Buscar modelo
        conn = sqlite3.connect('patterns.db')
        cursor = conn.cursor()
        cursor.execute('SELECT model_path, architecture FROM trained_models WHERE id = ?', (request.model_id,))
        row = cursor.fetchone()
        conn.close()

        if not row:
            raise HTTPException(status_code=404, detail="Modelo não encontrado")

        model_path, architecture_json = row
        architecture = json.loads(architecture_json)

        if not os.path.exists(model_path):
            raise HTTPException(status_code=404, detail="Arquivo do modelo não encontrado")

        # Carregar modelo HITL
        model = MLP_HITL(layer_sizes=architecture)
        model.load_model(model_path)

        # Converter padrão
        pattern_matrix = [[int(cell) for cell in row] for row in request.pattern]
        pattern_flat = np.array(pattern_matrix).flatten()

        # Predição com incerteza
        prediction, uncertainty, probabilities = model.predict_with_uncertainty(pattern_flat)

        # Converter probabilidades
        probs_dict = {}
        for label, idx in model.label_to_index.items():
            probs_dict[label] = float(probabilities[idx])

        # Determinar se precisa de feedback (Active Learning)
        needs_feedback = model.needs_feedback(uncertainty)

        return {
            "prediction": prediction,
            "uncertainty": float(uncertainty),
            "needs_feedback": bool(needs_feedback),
            "probabilities": probs_dict,
            "uncertainty_threshold": float(model.uncertainty_threshold)
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro na predição: {str(e)}")


@app.post("/api/submit-feedback")
async def submit_feedback(request: FeedbackRequest):
    """
    Recebe feedback humano e ajusta o modelo (RLHF).
    """
    try:
        # Buscar modelo
        conn = sqlite3.connect('patterns.db')
        cursor = conn.cursor()
        cursor.execute('SELECT model_path, architecture FROM trained_models WHERE id = ?', (request.model_id,))
        row = cursor.fetchone()

        if not row:
            conn.close()
            raise HTTPException(status_code=404, detail="Modelo não encontrado")

        model_path, architecture_json = row
        architecture = json.loads(architecture_json)

        if not os.path.exists(model_path):
            conn.close()
            raise HTTPException(status_code=404, detail="Arquivo do modelo não encontrado")

        # Carregar modelo HITL
        model = MLP_HITL(layer_sizes=architecture)
        model.load_model(model_path)

        # Converter padrão
        pattern_matrix = [[int(cell) for cell in row] for row in request.pattern]
        pattern_flat = np.array(pattern_matrix).flatten()
        pattern_str = ','.join([''.join(['1' if cell else '0' for cell in row]) for row in request.pattern])

        # Aplicar feedback (RLHF)
        result = model.learn_from_feedback(pattern_flat, request.correct_label, learning_rate_feedback=0.5)

        # Salvar modelo atualizado
        model.save_model(model_path)

        # Registrar feedback no banco
        cursor.execute('''
            INSERT INTO human_feedback (model_id, pattern, predicted_label, correct_label, uncertainty)
            VALUES (?, ?, ?, ?, ?)
        ''', (request.model_id, pattern_str, request.predicted_label, request.correct_label, request.uncertainty))

        conn.commit()
        conn.close()

        return {
            "success": result['success'],
            "improved": result.get('improved', False),
            "prediction_before": result.get('prediction_before'),
            "prediction_after": result.get('prediction_after'),
            "total_feedback": result.get('total_feedback_count', 0),
            "message": "Feedback aplicado com sucesso!"
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao processar feedback: {str(e)}")


@app.get("/api/feedback-stats/{model_id}")
async def get_feedback_stats(model_id: str):
    """
    Retorna estatísticas de feedback de um modelo.
    """
    try:
        # Buscar modelo
        conn = sqlite3.connect('patterns.db')
        cursor = conn.cursor()
        cursor.execute('SELECT model_path, architecture FROM trained_models WHERE id = ?', (model_id,))
        row = cursor.fetchone()

        if not row:
            conn.close()
            raise HTTPException(status_code=404, detail="Modelo não encontrado")

        model_path, architecture_json = row
        architecture = json.loads(architecture_json)

        # Carregar modelo
        model = MLP_HITL(layer_sizes=architecture)
        model.load_model(model_path)

        # Estatísticas do modelo
        model_stats = model.get_feedback_stats()

        # Estatísticas do banco de dados
        cursor.execute('SELECT COUNT(*) FROM human_feedback WHERE model_id = ?', (model_id,))
        db_feedback_count = cursor.fetchone()[0]

        cursor.execute('''
            SELECT correct_label, COUNT(*) as count
            FROM human_feedback
            WHERE model_id = ?
            GROUP BY correct_label
        ''', (model_id,))
        feedback_by_label = {row[0]: row[1] for row in cursor.fetchall()}

        conn.close()

        return {
            "model_id": model_id,
            "model_stats": model_stats,
            "db_feedback_count": db_feedback_count,
            "feedback_by_label": feedback_by_label
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao obter estatísticas: {str(e)}")
