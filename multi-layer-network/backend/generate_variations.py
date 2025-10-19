#!/usr/bin/env python3
"""
Gera variações de padrões deslocando-os pixel por pixel pelo grid.
"""
import numpy as np
import sqlite3
from datetime import datetime

DB_PATH = 'patterns.db'
GRID_SIZE = 16

def get_pattern_bounds(pattern):
    """Encontra os limites do padrão (bounding box)."""
    rows, cols = np.where(pattern == 1)
    if len(rows) == 0:
        return None

    min_row, max_row = rows.min(), rows.max()
    min_col, max_col = cols.min(), cols.max()

    return min_row, max_row, min_col, max_col

def extract_shape(pattern, bounds):
    """Extrai a forma do padrão dentro dos limites."""
    min_row, max_row, min_col, max_col = bounds
    return pattern[min_row:max_row+1, min_col:max_col+1]

def place_shape_at(shape, row_offset, col_offset):
    """Coloca a forma em uma posição específica do grid."""
    grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)

    shape_h, shape_w = shape.shape

    # Verificar se cabe
    if row_offset + shape_h > GRID_SIZE or col_offset + shape_w > GRID_SIZE:
        return None

    grid[row_offset:row_offset+shape_h, col_offset:col_offset+shape_w] = shape
    return grid

def generate_all_positions(shape, label, original_position=(0, 0)):
    """Gera todas as posições possíveis da forma no grid."""
    shape_h, shape_w = shape.shape
    variations = []

    max_row = GRID_SIZE - shape_h
    max_col = GRID_SIZE - shape_w

    print(f"  Forma: {shape_h}x{shape_w}")
    print(f"  Posições possíveis: {(max_row + 1) * (max_col + 1)}")
    print(f"  Posição original: {original_position} (será pulada)")

    # Percorrer todas as posições
    for row in range(max_row + 1):
        for col in range(max_col + 1):
            # Pular a posição original
            if (row, col) == original_position:
                continue

            new_pattern = place_shape_at(shape, row, col)
            if new_pattern is not None:
                variations.append((new_pattern, label))

    return variations

def main():
    print("=== Gerador de Variações ===\n")

    # Conectar ao banco
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Buscar padrões existentes
    cursor.execute('SELECT id, pattern, label FROM training_patterns ORDER BY id')
    rows = cursor.fetchall()

    if len(rows) == 0:
        print("⚠ Nenhum padrão encontrado no banco!")
        conn.close()
        return

    print(f"Padrões encontrados: {len(rows)}\n")

    all_variations = []

    for pattern_id, pattern_str, label in rows:
        print(f"Processando padrão #{pattern_id} (label: {label})")

        # Reconstruir matriz
        pattern_rows = pattern_str.split(',')
        pattern = np.array([[int(c) for c in row] for row in pattern_rows])

        # Encontrar limites
        bounds = get_pattern_bounds(pattern)
        if bounds is None:
            print("  ⚠ Padrão vazio, pulando...")
            continue

        # Extrair forma e posição original
        shape = extract_shape(pattern, bounds)
        original_position = (bounds[0], bounds[2])  # (min_row, min_col)

        # Forçar label como "No-T" para todas as variações
        target_label = "No-T"

        # Gerar variações (exceto a posição original)
        variations = generate_all_positions(shape, target_label, original_position)
        all_variations.extend(variations)

        print(f"  ✓ {len(variations)} variações geradas (original excluída)\n")

    # Adicionar variações
    print(f"Total de variações: {len(all_variations)}")
    print(f"Padrões originais: {len(rows)}")
    print(f"Novos padrões a adicionar: {len(all_variations)}")
    print("\nAdicionando variações ao banco...")

    added = 0
    for variation_pattern, variation_label in all_variations:
        # Converter para string
        pattern_str = ','.join([''.join(map(str, row)) for row in variation_pattern])

        # Inserir no banco
        cursor.execute(
            'INSERT INTO training_patterns (pattern, label) VALUES (?, ?)',
            (pattern_str, variation_label)
        )
        added += 1

        if added % 50 == 0:
            print(f"  {added}/{len(all_variations)} adicionadas...")

    conn.commit()
    print(f"\n✓ {added} variações adicionadas!")
    print(f"Total de padrões agora: {len(rows) + added}")

    conn.close()

if __name__ == '__main__':
    main()
