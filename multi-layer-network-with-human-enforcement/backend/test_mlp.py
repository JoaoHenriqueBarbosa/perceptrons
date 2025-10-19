#!/usr/bin/env python3
"""
Teste simples do MLP para verificar se o treinamento funciona.
"""
import numpy as np
from mlp import MLP

print("=== Teste do MLP ===\n")

# Criar dados sintéticos simples (XOR problem)
print("1. Criando dados de teste (problema XOR)...")
X_train = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

y_train = ['A', 'B', 'B', 'A']  # XOR: same -> A, different -> B

print(f"   - Amostras: {len(X_train)}")
print(f"   - Classes: {set(y_train)}")
print()

# Criar modelo
print("2. Criando MLP...")
layer_sizes = [2, 4, 2]  # 2 entradas -> 4 ocultos -> 2 saídas
print(f"   - Arquitetura: {' → '.join(map(str, layer_sizes))}")
model = MLP(layer_sizes=layer_sizes, learning_rate=0.5)
print()

# Treinar
print("3. Treinando...")
train_info = model.train(X_train, y_train, epochs=1000, batch_size=4)
print()

# Resultados
print("4. Resultados:")
print(f"   - Loss final: {train_info['final_loss']:.4f}")
print(f"   - Acurácia final: {train_info['final_accuracy']:.2%}")
print()

# Testar predições
print("5. Testando predições:")
for i, x in enumerate(X_train):
    pred = model.predict(x)
    true = y_train[i]
    probs = model.predict_proba(x)[0]
    print(f"   Input: {x} -> Predito: {pred} (Real: {true}) ✓" if pred == true else f"   Input: {x} -> Predito: {pred} (Real: {true}) ✗")

print("\n✓ Teste concluído!")
