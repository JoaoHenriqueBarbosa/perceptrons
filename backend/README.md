# Perceptrons API

API para salvar padrões de treinamento de perceptrons.

## Setup (primeira vez)

O ambiente virtual já está configurado! Se precisar recriar:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Rodar a API

Opção 1 - Usar o script:
```bash
./run.sh
```

Opção 2 - Manual:
```bash
source venv/bin/activate
uvicorn main:app --reload
```

A API estará disponível em `http://localhost:8000`

## Endpoints

- `POST /api/patterns` - Salvar um novo padrão
- `GET /api/patterns` - Listar todos os padrões salvos
- `GET /` - Status da API

## Banco de dados

Os padrões são salvos em SQLite no arquivo `patterns.db` (criado automaticamente).
