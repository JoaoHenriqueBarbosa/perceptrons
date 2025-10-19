# Perceptron - Neurônio Simples

Implementação de um perceptron (neurônio único) para classificação binária.

## O que é um Perceptron?

O perceptron é o modelo mais básico de rede neural, inventado por Frank Rosenblatt em 1957. Ele pode aprender a classificar padrões linearmente separáveis em duas classes.

### Como funciona:

1. **Entrada**: Recebe 256 valores (grid 16x16 achatada)
2. **Pesos**: Cada entrada tem um peso associado
3. **Soma ponderada**: `z = w₁x₁ + w₂x₂ + ... + w₂₅₆x₂₅₆ + bias`
4. **Ativação**: Se `z ≥ 0`, retorna 1 (classe positiva), senão 0 (classe negativa)

### Algoritmo de treinamento:

Para cada exemplo:
- Se predição está correta: não faz nada
- Se predição está errada: ajusta pesos na direção do erro
- Fórmula: `w = w + learning_rate × erro × x`

## Como usar

### 1. Preparar dados

Primeiro, desenhe e salve exemplos usando o frontend:
- Desenhe exemplos da letra T e salve (rotule como "T")
- Desenhe exemplos de outras letras e salve (rotule como "não-T" ou outra coisa)
- Recomendado: 10-20 exemplos de cada classe

### 2. Treinar o modelo

```bash
cd backend
source venv/bin/activate
python perceptron.py
```

O script irá:
1. Carregar padrões do banco de dados
2. Perguntar qual label treinar como classe positiva
3. Treinar o perceptron
4. Mostrar métricas de avaliação
5. Opcionalmente salvar o modelo

### 3. Usar o modelo treinado

```python
from perceptron import Perceptron
import numpy as np

# Carregar modelo salvo
model = Perceptron(input_size=256)
model.load_weights('perceptron_T.npz')

# Fazer predição em um novo padrão (grid 16x16)
pattern = np.array([[0, 1, 1, ...], ...])  # 16x16
pattern_flat = pattern.flatten()  # Achatar para 256 elementos

prediction = model.predict(pattern_flat)
# prediction = 1 (é T) ou 0 (não é T)
```

## API Python

### Classe Perceptron

#### `__init__(input_size, learning_rate=0.01)`
Cria um novo perceptron.
- `input_size`: Número de features (256 para grid 16x16)
- `learning_rate`: Taxa de aprendizado (0.01 é bom valor inicial)

#### `train(X_train, y_train, epochs=100)`
Treina o modelo.
- `X_train`: Array numpy (N × 256) com os padrões
- `y_train`: Array numpy (N,) com labels (0 ou 1)
- `epochs`: Número máximo de iterações
- Retorna: Lista com erros por época

#### `predict(X)`
Faz predição.
- `X`: Array numpy (256,) com um padrão
- Retorna: 0 ou 1

#### `save_weights(filepath)`
Salva modelo em arquivo .npz

#### `load_weights(filepath)`
Carrega modelo de arquivo .npz

### Funções auxiliares

#### `load_patterns_from_db(db_path='patterns.db')`
Carrega padrões do SQLite.
- Retorna: (lista de patterns, lista de labels)

#### `prepare_data(patterns, labels, positive_label)`
Prepara dados para classificação binária.
- `positive_label`: Label que será classe 1
- Retorna: (X, y) arrays numpy

#### `evaluate_model(perceptron, X_test, y_test)`
Avalia modelo.
- Retorna: Dicionário com accuracy, precision, recall, f1_score

## Métricas de Avaliação

- **Acurácia**: % de predições corretas
- **Precisão**: Dos que previ como T, quantos eram realmente T?
- **Recall**: Dos que eram T, quantos consegui identificar?
- **F1-Score**: Média harmônica de precisão e recall

## Limitações do Perceptron

❌ Não pode aprender padrões não-linearmente separáveis (ex: XOR)
✅ Funciona bem para problemas simples e linearmente separáveis
✅ Treina muito rápido
✅ Fácil de entender e implementar

## Exemplo de uso

```python
from perceptron import *

# Carregar dados
patterns, labels = load_patterns_from_db()

# Preparar para reconhecer "T"
X, y = prepare_data(patterns, labels, positive_label='T')

# Criar e treinar
model = Perceptron(input_size=256, learning_rate=0.01)
errors = model.train(X, y, epochs=100)

# Avaliar
metrics = evaluate_model(model, X, y)
print(f"Acurácia: {metrics['accuracy']:.2%}")

# Salvar
model.save_weights('perceptron_T.npz')
```

## Próximos passos

Depois de treinar o perceptron, você pode:
1. Criar uma página de teste no frontend
2. Integrar predição na API
3. Visualizar os pesos aprendidos
4. Testar com novos desenhos
