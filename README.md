# 🧠 Perceptrons - Evolução de Redes Neurais Educacionais

Sistema educacional progressivo de redes neurais, desde perceptrons simples até MLPs com Human-in-the-Loop (HITL).

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.12-blue.svg)
![React](https://img.shields.io/badge/react-18-blue.svg)

---

## 📚 Índice

1. [Visão Geral](#-visão-geral)
2. [Evolução do Projeto](#-evolução-do-projeto)
3. [Arquitetura](#-arquitetura)
4. [Instalação](#-instalação)
5. [Como Usar](#-como-usar)
6. [Conceitos Implementados](#-conceitos-implementados)
7. [Contribuindo](#-contribuindo)
8. [Licença](#-licença)

---

## 🎯 Visão Geral

Este projeto é uma jornada educacional completa através da evolução das redes neurais, implementada do zero com foco em **compreensão** e **interatividade**. Cada etapa do projeto representa um marco histórico na evolução das redes neurais.

### Por que este projeto existe?

- 📖 **Educacional**: Entender redes neurais implementando do zero
- 🔬 **Experimental**: Interface visual para testar conceitos
- 🎨 **Interativo**: Desenhe padrões e veja a rede aprender
- 🤖 **HITL**: Aprenda com feedback humano em tempo real

---

## 🚀 Evolução do Projeto

### Fase 1: Single-Layer Perceptron (1957)
**Baseado no Perceptron de Rosenblatt**

```
single-layer-perceptron/
```

**O que foi implementado:**
- ✅ Perceptron binário simples (função degrau)
- ✅ Codificação binária para múltiplas classes (N neurônios = 2^N classes)
- ✅ Algoritmo de aprendizado original de Rosenblatt
- ✅ Interface visual para desenhar padrões 16×16

**Limitações:**
- ❌ Apenas problemas linearmente separáveis
- ❌ Função de ativação não-diferenciável (degrau)
- ❌ Não pode aprender XOR ou padrões não-lineares

**Exemplo de uso:**
```bash
cd single-layer-perceptron/backend
./run.sh
```

**Aprendizado:**
- Convergência garantida para problemas linearmente separáveis
- Codificação binária criativa para múltiplas classes
- Base histórica das redes neurais

---

### Fase 2: Multi-Layer Perceptron (1986)
**Baseado em Rumelhart, Hinton e Williams**

```
multi-layer-network/
```

**O que foi implementado:**
- ✅ Múltiplas camadas (entrada → ocultas → saída)
- ✅ Função de ativação Sigmoid (diferenciável)
- ✅ **Backpropagation** completo
- ✅ Mini-batch gradient descent
- ✅ Xavier/Glorot initialization
- ✅ Gradient clipping
- ✅ Early stopping
- ✅ Cross-entropy loss

**Arquitetura:**
```python
# Exemplo: 256 → 64 → 32 → 2
MLP(layer_sizes=[256, 64, 32, 2], learning_rate=0.3)
```

**Melhorias sobre Single-Layer:**
- ✅ Aprende padrões não-lineares (XOR, círculos, etc)
- ✅ Representações hierárquicas
- ✅ Convergência mais robusta
- ✅ Generalização melhorada

**Dataset Balanceado:**
- 442 exemplos de "T" (3 variações de tamanho)
- 442 exemplos de "No-T" (8 letras + formas aleatórias)
- **882 amostras totais (50/50 balanceado)**

**Técnicas de Data Augmentation:**
```python
# Gera variações deslocando padrão pixel-by-pixel
# T de 5×5 → 144 posições (12×12 grid)
# T de 3×3 → 196 posições (14×14 grid)
# T de 7×7 → 100 posições (10×10 grid)
```

---

### Fase 3: Multi-Layer with Human-in-the-Loop (2023+)
**Baseado em RLHF (ChatGPT) e Active Learning**

```
multi-layer-network-with-human-enforcement/
```

**O que foi implementado:**

#### 1️⃣ **Active Learning**
Sistema pede feedback quando incerto (entropia > 30%)

```python
def predict_with_uncertainty(self, X):
    # Calcula entropia da distribuição de probabilidades
    entropy = -Σ(p * log(p))
    uncertainty = entropy / max_entropy

    if uncertainty > threshold:
        return "PEDIR FEEDBACK HUMANO"
```

**Quando pedir feedback:**
- 🟢 < 20% incerteza: Modelo confiante
- 🟠 20-40%: Incerteza média
- 🔴 > 40%: **Pedir feedback!**

#### 2️⃣ **RLHF (Reinforcement Learning from Human Feedback)**
Ajusta pesos instantaneamente com correção humana

```python
def learn_from_feedback(self, X, correct_label):
    # 1. Forward propagation
    prediction = self.predict(X)

    # 2. Backpropagation com label correto
    gradients = self.backprop(X, correct_label)

    # 3. Update com learning rate MAIOR (0.5)
    self.weights -= 0.5 * gradients

    # 4. Salvar modelo atualizado
    self.save_model()
```

**Diferença do treinamento normal:**
- LR maior (0.5 vs 0.1-0.3)
- Update imediato (não espera epoch)
- Salva modelo após cada feedback

#### 3️⃣ **Feedback Tracking**
Sistema monitora evolução com feedback humano

```python
# Estatísticas armazenadas
{
    'total_feedback': 45,
    'improvements': 38,
    'improvement_rate': 0.844,  # 84.4% das correções melhoraram
    'feedback_by_label': {
        'T': 22,
        'No-T': 23
    }
}
```

**Interface Visual:**
- 📊 Barra de incerteza colorida
- 🎯 Active Learning alerts
- 📈 Gráfico de probabilidades
- ✏️ Formulário de correção
- 📉 Estatísticas em tempo real

---

## 🏗️ Arquitetura

### Stack Tecnológico

**Backend:**
```
Python 3.12
├── FastAPI (API REST)
├── NumPy (Computação numérica)
├── SQLite (Persistência)
└── Uvicorn (Servidor ASGI)
```

**Frontend:**
```
React 18 + TypeScript
├── Vite (Build tool)
├── TanStack Router (Roteamento)
└── CSS Modules (Estilização)
```

### Estrutura de Diretórios

```
perceptrons/
│
├── single-layer-perceptron/
│   ├── backend/
│   │   ├── perceptron.py          # Perceptron simples
│   │   ├── multi_perceptron.py    # Codificação binária
│   │   ├── main.py                # API FastAPI
│   │   └── patterns.db            # Dataset
│   └── frontend/
│       └── src/
│           ├── pages/
│           │   ├── Dataset.tsx    # Criar padrões
│           │   ├── Training.tsx   # Treinar modelo
│           │   └── Test.tsx       # Testar modelo
│           └── App.tsx
│
├── multi-layer-network/
│   ├── backend/
│   │   ├── mlp.py                 # MLP com backprop
│   │   ├── main.py                # API REST
│   │   ├── patterns.db            # Dataset balanceado (882)
│   │   ├── generate_variations.py # Data augmentation
│   │   ├── build_balanced_dataset.py
│   │   └── models/                # Modelos treinados
│   └── frontend/
│       └── src/
│           └── pages/
│               ├── Training.tsx   # Config hiperparâmetros
│               └── Test.tsx
│
└── multi-layer-network-with-human-enforcement/
    ├── backend/
    │   ├── mlp.py                 # MLP base
    │   ├── mlp_hitl.py            # MLP + HITL ⭐
    │   ├── main.py                # API com endpoints HITL
    │   └── patterns.db            # Dataset + feedback
    └── frontend/
        └── src/
            └── pages/
                └── TestHITL.tsx   # Interface HITL ⭐
```

---

## 💻 Instalação

### Pré-requisitos

```bash
# Python 3.12+
python3 --version

# Node.js 18+
node --version
```

### Setup Rápido

**1. Clone o repositório:**
```bash
git clone https://github.com/seu-usuario/perceptrons.git
cd perceptrons
```

**2. Escolha qual versão rodar:**

#### Opção A: Single-Layer Perceptron
```bash
# Backend
cd single-layer-perceptron/backend
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
./run.sh  # ou: uvicorn main:app --reload

# Frontend (novo terminal)
cd ../frontend
npm install
npm run dev
```

#### Opção B: Multi-Layer Perceptron
```bash
# Backend
cd multi-layer-network/backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
./run.sh

# Frontend
cd ../frontend
npm install
npm run dev
```

#### Opção C: MLP with HITL ⭐
```bash
# Backend
cd multi-layer-network-with-human-enforcement/backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
./run.sh

# Frontend
cd ../frontend
npm install
npm run dev
```

**3. Acesse:**
- Frontend: http://localhost:5173
- API: http://localhost:8000
- Docs: http://localhost:8000/docs

---

## 🎮 Como Usar

### 1. Single-Layer Perceptron

**Passo a passo:**

1. **Dataset** (`/dataset`)
   - Desenhe padrões 16×16
   - Rotule como "T" ou outra letra
   - Salve múltiplos exemplos

2. **Training** (`/training`)
   - Escolha número de neurônios (1-4)
   - Configure épocas (padrão: 100)
   - Inicie treinamento
   - Veja codificação binária gerada

3. **Test** (`/test`)
   - Selecione modelo treinado
   - Desenhe padrão
   - Veja predição + código binário

### 2. Multi-Layer Perceptron

**Dataset Balanceado (já incluído):**
- ✅ 442 exemplos de "T"
- ✅ 442 exemplos de "No-T"
- ✅ Variações automáticas (data augmentation)

**Treinamento:**

```python
# Hiperparâmetros recomendados
{
    "hidden_layers": [64, 32],      # 256→64→32→2
    "learning_rate": 0.3,            # LR alto converge rápido
    "batch_size": 16,                # Mini-batch
    "epochs": 300                    # Early stopping ativo
}
```

**Resultados esperados:**
- Acurácia: 95-100%
- Convergência: ~100-200 épocas
- Loss final: < 0.05

### 3. MLP with Human-in-the-Loop

**Workflow:**

1. **Treinar modelo base** (`/training`)
   - Use dataset balanceado
   - Configure arquitetura (ex: 64,32)
   - Treine até ~95% acurácia

2. **Testar com HITL** (`/test-hitl`)
   - Desenhe padrão
   - Sistema mostra:
     - Predição
     - Incerteza (%)
     - Probabilidades por classe
     - Alert se incerto

3. **Fornecer Feedback**
   - Se predição errada → Corrigir
   - Sistema aplica RLHF imediatamente
   - Modelo salvo automaticamente
   - Veja estatísticas atualizadas

**Métricas HITL:**
```json
{
  "total_feedback": 50,
  "improvements": 42,
  "improvement_rate": 0.84,  // 84% melhoraram
  "feedback_by_label": {
    "T": 25,
    "No-T": 25
  }
}
```

---

## 📖 Conceitos Implementados

### Algoritmos

#### 1. Perceptron Learning Rule (1957)
```python
# Para cada exemplo (x, y):
prediction = sign(w·x + b)
if prediction != y:
    w = w + η * y * x
    b = b + η * y
```

#### 2. Backpropagation (1986)
```python
# Forward pass
for layer in layers:
    z = W @ a + b
    a = sigmoid(z)

# Backward pass
δ_L = (a_L - y) ⊙ σ'(z_L)
for layer in reversed(layers):
    ∇W = a^T @ δ / m
    ∇b = sum(δ) / m
    δ = (δ @ W^T) ⊙ σ'(a)

# Update
W = W - η * ∇W
b = b - η * ∇b
```

#### 3. Active Learning (2000s)
```python
# Entropy-based uncertainty
H(p) = -Σ p_i * log(p_i)
uncertainty = H(p) / log(K)  # Normalizado

if uncertainty > threshold:
    request_human_feedback()
```

#### 4. RLHF (2020s)
```python
# Recebe correção humana
correct_label = human_input()

# Backprop com label correto
loss = cross_entropy(prediction, correct_label)
gradients = backward(loss)

# Update com LR maior
weights -= 0.5 * gradients  # vs 0.1 normal
save_model()
```

### Técnicas de Otimização

| Técnica | Implementação | Benefício |
|---------|--------------|-----------|
| Xavier Init | `w ~ U[-√(6/(n_in+n_out)), +√(6/(n_in+n_out))]` | Evita vanishing/exploding gradients |
| Gradient Clipping | `if ‖∇w‖ > 5: ∇w = ∇w * (5/‖∇w‖)` | Estabiliza treinamento |
| Early Stopping | Para se val_loss não melhora por 50 épocas | Previne overfitting |
| Mini-batch | Batch size 16-32 | Balança velocidade e estabilidade |

### Data Augmentation

```python
# Gera variações deslocando padrão
def generate_variations(pattern):
    for row in range(max_positions_y):
        for col in range(max_positions_x):
            if (row, col) != original_position:
                yield shift_pattern(pattern, row, col)

# Exemplo: T de 5×5
# Grid 16×16 permite posições: (16-5+1) × (16-5+1) = 12×12 = 144
# Menos posição original = 143 variações
```

---

## 📊 Resultados

### Single-Layer Perceptron

**Problema: Classificar "T" vs "No-T"**

| Métrica | Valor |
|---------|-------|
| Dataset | 62 exemplos (manual) |
| Acurácia | ~70-80% |
| Limitação | Aprende apenas 1 feature (ex: "pixels no topo") |

**Conclusão:** Insuficiente para padrões complexos.

### Multi-Layer Perceptron

**Mesmo problema com dataset balanceado**

| Métrica | Valor |
|---------|-------|
| Dataset | 882 exemplos (balanceado 50/50) |
| Arquitetura | 256→64→32→2 |
| Acurácia | **95-100%** ✅ |
| Loss final | 0.015-0.050 |
| Épocas | ~100-200 (early stop) |

**Conclusão:** Aprende representações complexas!

### MLP with HITL

**Evolução contínua com feedback**

| Métrica | Inicial | Após 50 Feedbacks |
|---------|---------|-------------------|
| Acurácia | 95% | **98-99%** |
| Casos incertos | 15% | **5%** |
| Taxa de melhoria | - | 84% |

**Conclusão:** Feedback humano elimina erros residuais!

---

## 🎓 Aprendizados

### 1. Por que Single-Layer falha?

**Problema XOR:**
```
Input: (0,0) → 0
Input: (0,1) → 1
Input: (1,0) → 1
Input: (1,1) → 0
```

Não existe linha reta que separe estas classes! Perceptron simples não consegue.

### 2. Como MLP resolve?

**Camadas ocultas criam representações:**
```
Camada 1: Detecta bordas
Camada 2: Detecta formas
Camada 3: Detecta letras
```

Transforma problema não-linear em linear no espaço de features.

### 3. Por que HITL melhora?

**Active Learning:**
- Foca em exemplos difíceis
- Humano fornece conhecimento onde modelo falha

**RLHF:**
- Correção instantânea
- Aprende padrões que treinamento offline perdeu

---

## 🛠️ Scripts Úteis

### Data Augmentation
```bash
cd multi-layer-network/backend
python3 generate_variations.py
# Gera variações de padrões deslocados
```

### Construir Dataset Balanceado
```bash
python3 build_balanced_dataset.py
# Combina Ts + No-Ts em proporção 50/50
```

### Analisar Dataset
```bash
python3 analyze_dataset.py
# Mostra distribuição, variância, separabilidade
```

### Testar MLP Standalone
```bash
python3 mlp.py
# Interface CLI para treinar e testar
```

### Testar HITL Standalone
```bash
python3 mlp_hitl.py
# Demo de Active Learning + RLHF
```

---

## 🐛 Troubleshooting

### Backend não inicia

```bash
# Verificar Python
python3 --version  # Deve ser 3.12+

# Recriar venv
rm -rf venv
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Frontend erro de CORS

```bash
# Verificar se backend está rodando
curl http://localhost:8000/

# Verificar porta correta
# Backend: 8000
# Frontend: 5173
```

### Modelo não converge

```python
# Ajustar hiperparâmetros
{
    "learning_rate": 0.3,      # Aumentar se muito lento
    "batch_size": 16,          # Diminuir se instável
    "hidden_layers": [64, 32]  # Simplificar arquitetura
}
```

### Dataset desbalanceado

```bash
# Reconstruir dataset balanceado
python3 build_balanced_dataset.py
```

---

## 📚 Referências

### Papers Clássicos

1. **Rosenblatt, F. (1957)**
   - "The Perceptron: A Probabilistic Model for Information Storage and Organization in the Brain"
   - Psychological Review

2. **Minsky & Papert (1969)**
   - "Perceptrons" (livro que mostrou limitações)

3. **Rumelhart, Hinton & Williams (1986)**
   - "Learning representations by back-propagating errors"
   - Nature

4. **Glorot & Bengio (2010)**
   - "Understanding the difficulty of training deep feedforward neural networks"
   - AISTATS

### Conceitos Modernos

5. **Christiano et al. (2017)**
   - "Deep Reinforcement Learning from Human Preferences"
   - NeurIPS

6. **Ouyang et al. (2022)**
   - "Training language models to follow instructions with human feedback" (ChatGPT)
   - OpenAI

---

## 🤝 Contribuindo

Contribuições são bem-vindas! Veja [CONTRIBUTING.md](CONTRIBUTING.md)

**Ideias para contribuir:**

- 🎨 Novos datasets (números, formas, etc)
- 🧪 Funções de ativação (ReLU, tanh, etc)
- 📊 Visualização de pesos/ativações
- 🎯 Novos algoritmos (Dropout, Batch Norm, etc)
- 📱 Interface mobile
- 🌐 Outros idiomas

---

## 📄 Licença

MIT License - veja [LICENSE](LICENSE)

---

## 👏 Agradecimentos

- Frank Rosenblatt - Inventor do Perceptron
- Geoffrey Hinton - Pai do Deep Learning
- Yann LeCun - Pioneiro em CNNs
- OpenAI - Inspiração para RLHF

---

## 📬 Contato

**Projeto:** Sistema Educacional de Perceptrons
**Objetivo:** Ensinar redes neurais através de implementação prática

---

<p align="center">
  Feito com ❤️ e 🧠 por estudantes, para estudantes
</p>

<p align="center">
  <i>"The best way to learn is to build" - Andrew Ng</i>
</p>
