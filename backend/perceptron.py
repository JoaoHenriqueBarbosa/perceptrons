import numpy as np
import sqlite3
from typing import List, Tuple


class Perceptron:
    """
    Implementação de um Perceptron simples (neurônio único).

    O perceptron é o modelo mais básico de rede neural, capaz de aprender
    funções linearmente separáveis.
    """

    def __init__(self, input_size: int, learning_rate: float = 0.01):
        """
        Inicializa o perceptron.

        Args:
            input_size: Número de features de entrada (256 para grid 16x16)
            learning_rate: Taxa de aprendizado (default: 0.01)
        """
        # Inicializar pesos aleatórios pequenos
        self.weights = np.random.randn(input_size) * 0.01
        # Inicializar bias
        self.bias = 0.0
        self.learning_rate = learning_rate

    def activation(self, x: float) -> int:
        """
        Função de ativação degrau (step function).

        Args:
            x: Valor de entrada

        Returns:
            1 se x >= 0, caso contrário 0
        """
        return 1 if x >= 0 else 0

    def predict(self, X: np.ndarray) -> int:
        """
        Faz a predição para uma entrada.

        Args:
            X: Array de features (1D array com 256 elementos)

        Returns:
            Classe predita (0 ou 1)
        """
        # Calcular soma ponderada: w·x + b
        weighted_sum = np.dot(self.weights, X) + self.bias
        # Aplicar função de ativação
        return self.activation(weighted_sum)

    def train(self, X_train: np.ndarray, y_train: np.ndarray, epochs: int = 100) -> List[float]:
        """
        Treina o perceptron usando o algoritmo de aprendizado do perceptron.

        Args:
            X_train: Matriz de features (N x 256)
            y_train: Array de labels (N elementos, 0 ou 1)
            epochs: Número de épocas de treinamento

        Returns:
            Lista com o erro em cada época
        """
        errors = []

        for epoch in range(epochs):
            total_error = 0

            # Para cada exemplo de treinamento
            for i in range(len(X_train)):
                # Fazer predição
                prediction = self.predict(X_train[i])

                # Calcular erro
                error = y_train[i] - prediction

                # Atualizar pesos se houver erro
                if error != 0:
                    # Regra de aprendizado do perceptron:
                    # w = w + learning_rate * error * x
                    self.weights += self.learning_rate * error * X_train[i]
                    self.bias += self.learning_rate * error
                    total_error += abs(error)

            errors.append(total_error)

            # Se não houver erros, o modelo convergiu
            if total_error == 0:
                print(f"Convergiu na época {epoch + 1}")
                break

        return errors

    def save_weights(self, filepath: str):
        """
        Salva os pesos do modelo em um arquivo.

        Args:
            filepath: Caminho do arquivo para salvar
        """
        np.savez(filepath, weights=self.weights, bias=self.bias)
        print(f"Modelo salvo em {filepath}")

    def load_weights(self, filepath: str):
        """
        Carrega os pesos do modelo de um arquivo.

        Args:
            filepath: Caminho do arquivo para carregar
        """
        data = np.load(filepath)
        self.weights = data['weights']
        self.bias = data['bias']
        print(f"Modelo carregado de {filepath}")


def load_patterns_from_db(db_path: str = 'patterns.db') -> Tuple[List[np.ndarray], List[str]]:
    """
    Carrega os padrões do banco de dados SQLite.

    Args:
        db_path: Caminho do banco de dados

    Returns:
        Tupla com (lista de patterns, lista de labels)
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('SELECT pattern, label FROM training_patterns')
    rows = cursor.fetchall()
    conn.close()

    patterns = []
    labels = []

    for pattern_str, label in rows:
        # Reconstruir a matriz 16x16
        pattern_rows = pattern_str.split(',')
        pattern_matrix = [[int(c) for c in row] for row in pattern_rows]
        # Achatar para 1D array (256 elementos)
        pattern_flat = np.array(pattern_matrix).flatten()
        patterns.append(pattern_flat)
        labels.append(label)

    return patterns, labels


def prepare_data(patterns: List[np.ndarray], labels: List[str],
                 positive_label: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepara os dados para treinamento binário.

    Args:
        patterns: Lista de padrões (arrays 1D)
        labels: Lista de labels (strings)
        positive_label: Label que será considerado como classe positiva (1)

    Returns:
        Tupla com (X, y) onde X são os patterns e y são labels binários (0 ou 1)
    """
    X = np.array(patterns)
    # Converter labels para binário: 1 se for positive_label, 0 caso contrário
    y = np.array([1 if label == positive_label else 0 for label in labels])

    return X, y


def evaluate_model(perceptron: Perceptron, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """
    Avalia o modelo no conjunto de teste.

    Args:
        perceptron: Modelo treinado
        X_test: Padrões de teste
        y_test: Labels de teste

    Returns:
        Dicionário com métricas de avaliação
    """
    predictions = [perceptron.predict(x) for x in X_test]

    # Calcular métricas
    correct = sum([1 for pred, true in zip(predictions, y_test) if pred == true])
    accuracy = correct / len(y_test) if len(y_test) > 0 else 0

    # Calcular precisão, recall e F1 para classe positiva
    true_positives = sum([1 for pred, true in zip(predictions, y_test) if pred == 1 and true == 1])
    false_positives = sum([1 for pred, true in zip(predictions, y_test) if pred == 1 and true == 0])
    false_negatives = sum([1 for pred, true in zip(predictions, y_test) if pred == 0 and true == 1])

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'correct': correct,
        'total': len(y_test)
    }


if __name__ == '__main__':
    """
    Exemplo de uso do perceptron.
    """
    print("=== Perceptron Trainer ===\n")

    # Carregar dados do banco
    print("Carregando dados do banco...")
    patterns, labels = load_patterns_from_db()

    if len(patterns) == 0:
        print("Nenhum padrão encontrado no banco de dados!")
        print("Por favor, desenhe e salve alguns exemplos primeiro.")
        exit(1)

    print(f"Carregados {len(patterns)} padrões")
    print(f"Labels únicos: {set(labels)}\n")

    # Preparar dados (exemplo: treinar para reconhecer "T")
    # Você pode mudar o positive_label para o que quiser treinar
    positive_label = input("Qual label você quer treinar como classe positiva? (ex: 'T'): ").strip()

    if positive_label not in labels:
        print(f"Label '{positive_label}' não encontrado nos dados!")
        exit(1)

    X, y = prepare_data(patterns, labels, positive_label)

    print(f"\nDados preparados:")
    print(f"- Classe positiva (1): {positive_label}")
    print(f"- Total de exemplos: {len(X)}")
    print(f"- Exemplos positivos: {sum(y)}")
    print(f"- Exemplos negativos: {len(y) - sum(y)}\n")

    # Criar e treinar perceptron
    print("Criando perceptron...")
    perceptron = Perceptron(input_size=256, learning_rate=0.01)

    print("Treinando...\n")
    errors = perceptron.train(X, y, epochs=100)

    print(f"\nTreinamento concluído!")
    print(f"Épocas executadas: {len(errors)}")
    print(f"Erro final: {errors[-1]}")

    # Avaliar no conjunto de treinamento (idealmente você teria um conjunto de teste)
    print("\nAvaliação no conjunto de treinamento:")
    metrics = evaluate_model(perceptron, X, y)
    print(f"Acurácia: {metrics['accuracy']:.2%}")
    print(f"Precisão: {metrics['precision']:.2%}")
    print(f"Recall: {metrics['recall']:.2%}")
    print(f"F1-Score: {metrics['f1_score']:.2%}")
    print(f"Corretos: {metrics['correct']}/{metrics['total']}")

    # Salvar modelo
    save = input("\nDeseja salvar o modelo? (s/n): ").strip().lower()
    if save == 's':
        perceptron.save_weights(f'perceptron_{positive_label}.npz')
