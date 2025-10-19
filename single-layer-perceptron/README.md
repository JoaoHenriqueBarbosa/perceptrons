# Perceptrons - Sistema Educacional de Redes Neurais

![Python](https://img.shields.io/badge/Python-3.12-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115.0-green)
![React](https://img.shields.io/badge/React-18-61dafb)
![TypeScript](https://img.shields.io/badge/TypeScript-5-blue)

Um sistema completo e interativo para aprender sobre **Perceptrons** (neurônios simples) através de experimentação prática. Desenhe padrões, treine modelos e veja as limitações e capacidades de redes neurais básicas em tempo real.

## 📚 O que são Perceptrons?

O **Perceptron** é o modelo mais básico de rede neural, inventado por Frank Rosenblatt em 1957. É um classificador linear que aprende a separar padrões em classes usando uma simples regra de aprendizado.

### Como Funciona:

```
Entrada (256 pixels) → Pesos → Soma Ponderada → Ativação (0 ou 1)
```

- **Entrada**: Grid 16×16 achatada (256 valores)
- **Pesos**: Um peso para cada pixel
- **Soma**: `z = w₁x₁ + w₂x₂ + ... + w₂₅₆x₂₅₆ + bias`
- **Ativação**: Se `z ≥ 0` → Classe 1, senão → Classe 0

### Múltiplos Neurônios (Codificação Binária):

Com **N neurônios**, podemos representar até **2^N classes**:
- 1 neurônio: 2 classes (0, 1)
- 2 neurônios: 4 classes (00, 01, 10, 11)
- 3 neurônios: 8 classes (000, 001, 010, 011, 100, 101, 110, 111)

Cada neurônio aprende uma "pergunta binária", e a combinação identifica a classe!

## 🎯 O que você vai aprender:

### ✅ Conceitos que FUNCIONAM:
- Perceptrons conseguem separar classes **linearmente separáveis**
- Com dados suficientes (10-20 exemplos/classe), convergem rapidamente
- Múltiplos neurônios podem codificar múltiplas classes
- A importância de **balanceamento de classes** no dataset

### ❌ Limitações descobertas na prática:
- Perceptrons **NÃO** aprendem padrões não-lineares (ex: XOR)
- Classes muito similares (T, H, L) são difíceis de distinguir
- Com apenas 1 classe no dataset, o modelo não aprende nada útil
- Fronteiras lineares são muito simples para problemas complexos

### 🧠 Insights importantes:
- **Dados > Algoritmo**: Com bons dados, até modelos simples funcionam
- **Experimentação é aprendizado**: Ver o modelo errar ensina tanto quanto vê-lo acertar
- **Trade-offs**: Mais neurônios = mais classes, mas também mais complexidade

## 🚀 Como Rodar o Projeto

### Pré-requisitos

- **Python 3.12+**
- **Node.js 18+** ou **Bun**
- **Git**

### 1. Clone o repositório

```bash
git clone https://github.com/seu-usuario/perceptrons.git
cd perceptrons
```

### 2. Backend (API Python)

```bash
cd backend

# Criar ambiente virtual
python3 -m venv venv

# Ativar ambiente virtual
source venv/bin/activate  # Linux/Mac
# OU
venv\Scripts\activate  # Windows

# Instalar dependências
pip install -r requirements.txt

# Rodar a API
uvicorn main:app --reload
```

A API estará em: `http://localhost:8000`

### 3. Frontend (React + TypeScript)

Em outro terminal:

```bash
cd frontend

# Instalar dependências
bun install  # ou npm install

# Rodar o frontend
bun run dev  # ou npm run dev
```

O frontend estará em: `http://localhost:5173`

## 📖 Como Usar

### 1. Dataset - Criar Exemplos de Treinamento

**Acesse:** `http://localhost:5173/dataset`

1. **Digite um label** (classe) no campo: `T`, `não-T`, `H`, `L`, etc.
2. **Desenhe o padrão** na grid 16×16:
   - Clique e arraste para desenhar
   - Só pinta pixels (não apaga)
3. **Salve o exemplo** - Grid limpa automaticamente
4. **Repita** 10-20 vezes por classe

**Dicas:**
- Crie **variações** (T largo, T fino, T torto)
- Mantenha **balanceamento** (mesma quantidade por classe)
- Use o botão **"Atualizar Labels sem classe"** para corrigir labels em massa

**Estatísticas:** Veja quantos exemplos você tem de cada classe!

### 2. Training - Treinar Modelos

**Acesse:** `http://localhost:5173/training`

1. **Configure o número de neurônios:**
   - 1 neurônio = 2 classes (binário)
   - 2 neurônios = 4 classes
   - 3 neurônios = 8 classes
   - 4 neurônios = 16 classes

2. **Configure as épocas:** 10-500 iterações
   - Modelo **converge** quando erro = 0
   - Se convergir rápido (ex: 2 épocas), está aprendendo bem!

3. **Clique "Iniciar Treinamento"**

4. **Veja o resultado:**
   - Acurácia
   - Épocas executadas
   - Convergência
   - Número de classes detectadas

**Modelos são salvos automaticamente** e aparecem na lista!

### 3. Test - Testar Padrões

**Acesse:** `http://localhost:5173/test`

1. **Selecione um modelo treinado**
2. **Desenhe um padrão de teste**
3. **Clique "Testar Padrão"**
4. **Veja a predição:**
   - Classe identificada
   - Código binário dos neurônios

## 🔬 Experimentos Recomendados

### Experimento 1: Classificação Binária Simples
**Objetivo:** Ver o Perceptron funcionando perfeitamente

1. Crie 2 classes **muito diferentes**:
   - **Classe "T"**: 15 exemplos da letra T
   - **Classe "vazio"**: 15 exemplos de grids vazias

2. Treine com **1 neurônio**, **100 épocas**

3. Teste: Desenhe T → retorna "T", Desenhe vazio → retorna "vazio"

**Resultado esperado:** Acurácia ~100%, convergência rápida ✓

### Experimento 2: Múltiplas Classes Distintas
**Objetivo:** Ver codificação binária em ação

1. Crie 4 classes distintas:
   - Linha vertical
   - Linha horizontal
   - Diagonal /
   - Diagonal \

2. Treine com **2 neurônios** (2² = 4 classes)

3. Veja o mapeamento:
   - Vertical → 00
   - Horizontal → 01
   - Diagonal / → 10
   - Diagonal \ → 11

**Resultado esperado:** Acurácia ~80-90% ✓

### Experimento 3: Classes Similares (Limitação)
**Objetivo:** Ver onde o Perceptron falha

1. Crie 3 classes similares:
   - **T**: 20 exemplos
   - **H**: 20 exemplos
   - **L**: 20 exemplos

2. Treine com **2 neurônios**

3. Teste: Provavelmente vai confundir as letras!

**Resultado esperado:** Acurácia ~40-60%, muita confusão ✗

**Por quê?** T, H, L são muito parecidos. Perceptrons só aprendem fronteiras lineares!

### Experimento 4: Dataset Desbalanceado
**Objetivo:** Ver importância do balanceamento

1. Crie:
   - **T**: 50 exemplos
   - **não-T**: 5 exemplos

2. Treine com **1 neurônio**

3. Teste: Modelo vai prever "T" para quase tudo!

**Resultado esperado:** Alta acurácia aparente, mas modelo inútil ✗

## 📊 Estrutura do Projeto

```
perceptrons/
├── backend/
│   ├── main.py                 # API FastAPI
│   ├── perceptron.py           # Classe Perceptron
│   ├── multi_perceptron.py     # Múltiplos neurônios
│   ├── requirements.txt        # Dependências Python
│   ├── patterns.db            # SQLite (criado automaticamente)
│   └── models/                # Modelos treinados (criado automaticamente)
│
├── frontend/
│   ├── src/
│   │   ├── pages/
│   │   │   ├── Home.tsx       # Página inicial
│   │   │   ├── Dataset.tsx    # Criar dataset
│   │   │   ├── Training.tsx   # Treinar modelos
│   │   │   └── Test.tsx       # Testar padrões
│   │   ├── routes/
│   │   │   └── router.tsx     # TanStack Router
│   │   ├── App.tsx            # App principal
│   │   └── App.css            # Estilos
│   ├── package.json
│   └── vite.config.ts
│
└── README.md
```

## 🔧 Tecnologias Utilizadas

### Backend:
- **FastAPI** - Framework web moderno e rápido
- **NumPy** - Computação numérica
- **SQLite** - Banco de dados leve
- **Uvicorn** - Servidor ASGI

### Frontend:
- **React 18** - UI Library
- **TypeScript** - Tipagem estática
- **TanStack Router** - Roteamento
- **Vite** - Build tool
- **Bun** - Runtime JavaScript

## 📝 API Endpoints

### Padrões
- `POST /api/patterns` - Salvar novo padrão
- `GET /api/patterns` - Listar todos os padrões
- `DELETE /api/patterns/{id}` - Deletar padrão
- `PUT /api/patterns/bulk-update-labels` - Atualizar labels em massa

### Treinamento
- `POST /api/train` - Treinar novo modelo
- `GET /api/models` - Listar modelos treinados

### Predição
- `POST /api/predict` - Fazer predição com modelo

## 🎓 Conceitos Aprendidos

### 1. Algoritmo de Aprendizado do Perceptron

```python
for epoch in range(epochs):
    for cada exemplo (x, y):
        predição = predict(x)
        erro = y - predição

        if erro != 0:
            # Ajustar pesos
            w = w + learning_rate × erro × x
            b = b + learning_rate × erro
```

**Intuição:** Se errou, ajusta os pesos na direção correta!

### 2. Separabilidade Linear

**Pode ser separado com Perceptron:**
```
Classe A: ●●●     |     Classe B:     ○○○
```

**NÃO pode ser separado com Perceptron (XOR):**
```
● ○
○ ●
```

Precisa de fronteira não-linear → Perceptron simples falha!

### 3. Convergência

- Se classes são linearmente separáveis → **sempre converge**
- Se não são → **nunca converge** (oscila indefinidamente)
- Número de épocas até convergência depende da complexidade

### 4. Codificação Binária para Múltiplas Classes

Com 2 neurônios classificando 4 classes:

```
Neurônio 1    Neurônio 2    Classe
    0             0           A
    0             1           B
    1             0           C
    1             1           D
```

Cada neurônio aprende uma "pergunta" diferente!

## ⚠️ Limitações Conhecidas

### Do Perceptron:
- ❌ Não aprende padrões não-lineares
- ❌ Sensível a outliers
- ❌ Pode não convergir com dados não-separáveis
- ❌ Fronteiras de decisão muito simples

### Do Projeto:
- Grid fixa 16×16 (256 features)
- Sem pré-processamento de imagem
- Sem validação cruzada
- Sem regularização
- Treinamento no conjunto completo (sem train/test split)

## 🚀 Melhorias Futuras

- [ ] Implementar MLP (Multi-Layer Perceptron) com backpropagation
- [ ] Adicionar visualização dos pesos aprendidos
- [ ] Train/test split automático
- [ ] Gráficos de erro por época
- [ ] Export de modelos para uso externo
- [ ] Importar imagens externas
- [ ] Data augmentation
- [ ] Métricas mais detalhadas (matriz de confusão, precisão, recall)

## 🤝 Contribuindo

Contribuições são bem-vindas! Este é um projeto educacional, então fique à vontade para:

1. Fork o projeto
2. Crie uma branch (`git checkout -b feature/AmazingFeature`)
3. Commit suas mudanças (`git commit -m 'Add some AmazingFeature'`)
4. Push para a branch (`git push origin feature/AmazingFeature`)
5. Abra um Pull Request

## 📄 Licença

MIT License - Sinta-se livre para usar este projeto para aprender e ensinar!

## 🙏 Agradecimentos

- Frank Rosenblatt - Por inventar o Perceptron em 1957
- Comunidade Python e React por ferramentas incríveis
- Você, por se interessar em aprender sobre redes neurais! 🎉

## 📚 Recursos Adicionais

- [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/) - Michael Nielsen
- [The Perceptron](https://en.wikipedia.org/wiki/Perceptron) - Wikipedia
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [React Documentation](https://react.dev/)

---

**Feito com ❤️ para educação em Machine Learning**

*"A melhor forma de aprender é experimentar!"*
