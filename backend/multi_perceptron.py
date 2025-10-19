import numpy as np
import sqlite3
import json
from typing import List, Tuple, Dict
from perceptron import Perceptron


class MultiPerceptron:
    """
    Múltiplos perceptrons para classificação multi-classe usando codificação binária.

    Com N neurônios, podemos representar até 2^N classes:
    - 1 neurônio: 2 classes (0, 1)
    - 2 neurônios: 4 classes (00, 01, 10, 11)
    - 3 neurônios: 8 classes (000, 001, 010, 011, 100, 101, 110, 111)
    """

    def __init__(self, input_size: int, num_neurons: int, learning_rate: float = 0.01):
        """
        Inicializa múltiplos perceptrons.

        Args:
            input_size: Tamanho da entrada (256 para grid 16x16)
            num_neurons: Número de neurônios/perceptrons
            learning_rate: Taxa de aprendizado
        """
        self.num_neurons = num_neurons
        self.input_size = input_size
        self.learning_rate = learning_rate

        # Criar N perceptrons
        self.neurons = [Perceptron(input_size, learning_rate) for _ in range(num_neurons)]

        # Mapear labels string para códigos binários
        self.label_to_code = {}
        self.code_to_label = {}

    def encode_labels(self, labels: List[str]) -> Dict[str, List[int]]:
        """
        Cria codificação binária para os labels únicos.

        Args:
            labels: Lista de labels (strings)

        Returns:
            Dicionário mapeando label -> código binário
        """
        unique_labels = sorted(set(labels))
        max_classes = 2 ** self.num_neurons

        if len(unique_labels) > max_classes:
            raise ValueError(
                f"Número de classes ({len(unique_labels)}) excede capacidade "
                f"de {self.num_neurons} neurônios ({max_classes} classes)"
            )

        # Atribuir código binário para cada label
        for i, label in enumerate(unique_labels):
            # Converter índice para binário com N bits
            binary_code = [int(x) for x in format(i, f'0{self.num_neurons}b')]
            self.label_to_code[label] = binary_code
            self.code_to_label[tuple(binary_code)] = label

        return self.label_to_code

    def train(self, X_train: np.ndarray, y_train: List[str], epochs: int = 100) -> Dict:
        """
        Treina todos os neurônios.

        Args:
            X_train: Padrões de entrada (N × 256)
            y_train: Labels (lista de strings)
            epochs: Número de épocas

        Returns:
            Dicionário com informações do treinamento
        """
        # Criar codificação
        self.encode_labels(y_train)

        # Converter labels para códigos binários
        y_encoded = np.array([self.label_to_code[label] for label in y_train])

        # Treinar cada neurônio independentemente
        all_errors = []
        for i, neuron in enumerate(self.neurons):
            print(f"Treinando neurônio {i + 1}/{self.num_neurons}...")
            # Cada neurônio aprende um bit da codificação
            y_bit = y_encoded[:, i]
            errors = neuron.train(X_train, y_bit, epochs)
            all_errors.append(errors)

        # Calcular época de convergência (quando todos convergem)
        convergence_epoch = max(len(errors) for errors in all_errors)
        converged = all(errors[-1] == 0 for errors in all_errors)

        return {
            'neurons': self.num_neurons,
            'epochs_executed': convergence_epoch,
            'convergence': converged,
            'label_mapping': self.label_to_code,
            'all_errors': all_errors
        }

    def predict(self, X: np.ndarray) -> str:
        """
        Faz predição para uma entrada.

        Args:
            X: Padrão de entrada (256 elementos)

        Returns:
            Label predito (string)
        """
        # Obter predição de cada neurônio
        binary_code = tuple(neuron.predict(X) for neuron in self.neurons)

        # Converter código binário de volta para label
        return self.code_to_label.get(binary_code, "desconhecido")

    def evaluate(self, X_test: np.ndarray, y_test: List[str]) -> Dict:
        """
        Avalia o modelo.

        Args:
            X_test: Padrões de teste
            y_test: Labels de teste

        Returns:
            Dicionário com métricas
        """
        predictions = [self.predict(x) for x in X_test]

        correct = sum(1 for pred, true in zip(predictions, y_test) if pred == true)
        accuracy = correct / len(y_test) if len(y_test) > 0 else 0

        # Contar predições por classe
        class_counts = {}
        for pred, true in zip(predictions, y_test):
            if true not in class_counts:
                class_counts[true] = {'correct': 0, 'total': 0, 'predicted': 0}

            class_counts[true]['total'] += 1
            if pred == true:
                class_counts[true]['correct'] += 1

        for pred in predictions:
            if pred in class_counts:
                class_counts[pred]['predicted'] += 1

        return {
            'accuracy': accuracy,
            'correct': correct,
            'total': len(y_test),
            'predictions': predictions,
            'class_counts': class_counts
        }

    def save_model(self, filepath: str):
        """
        Salva o modelo completo.

        Args:
            filepath: Caminho do arquivo
        """
        model_data = {
            'num_neurons': self.num_neurons,
            'input_size': self.input_size,
            'learning_rate': self.learning_rate,
            'label_to_code': self.label_to_code,
            'code_to_label': {str(k): v for k, v in self.code_to_label.items()},
            'weights': [neuron.weights.tolist() for neuron in self.neurons],
            'biases': [neuron.bias for neuron in self.neurons]
        }

        with open(filepath, 'w') as f:
            json.dump(model_data, f)

        print(f"Modelo salvo em {filepath}")

    def load_model(self, filepath: str):
        """
        Carrega o modelo completo.

        Args:
            filepath: Caminho do arquivo
        """
        with open(filepath, 'r') as f:
            model_data = json.load(f)

        self.num_neurons = model_data['num_neurons']
        self.input_size = model_data['input_size']
        self.learning_rate = model_data['learning_rate']
        self.label_to_code = model_data['label_to_code']
        self.code_to_label = {eval(k): v for k, v in model_data['code_to_label'].items()}

        # Recriar neurônios com pesos carregados
        self.neurons = []
        for weights, bias in zip(model_data['weights'], model_data['biases']):
            neuron = Perceptron(self.input_size, self.learning_rate)
            neuron.weights = np.array(weights)
            neuron.bias = bias
            self.neurons.append(neuron)

        print(f"Modelo carregado de {filepath}")


def load_patterns_from_db(db_path: str = 'patterns.db') -> Tuple[np.ndarray, List[str]]:
    """
    Carrega padrões do banco de dados.

    Returns:
        Tupla (X, y) onde X são os patterns e y são os labels
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('SELECT pattern, label FROM training_patterns')
    rows = cursor.fetchall()
    conn.close()

    if len(rows) == 0:
        raise ValueError("Nenhum padrão encontrado no banco de dados!")

    patterns = []
    labels = []

    for pattern_str, label in rows:
        # Reconstruir matriz 16x16
        pattern_rows = pattern_str.split(',')
        pattern_matrix = [[int(c) for c in row] for row in pattern_rows]
        pattern_flat = np.array(pattern_matrix).flatten()
        patterns.append(pattern_flat)
        labels.append(label if label else "sem_label")

    return np.array(patterns), labels


if __name__ == '__main__':
    """
    Exemplo de uso do MultiPerceptron.
    """
    print("=== Multi-Perceptron Trainer ===\n")

    # Carregar dados
    X, y = load_patterns_from_db()

    print(f"Dados carregados:")
    print(f"- Total de exemplos: {len(X)}")
    print(f"- Classes únicas: {set(y)}")
    print(f"- Número de classes: {len(set(y))}\n")

    # Configurar treinamento
    num_neurons = int(input("Quantos neurônios usar? (1-4): "))
    epochs = int(input("Quantas épocas? (padrão 100): ") or "100")

    max_classes = 2 ** num_neurons
    print(f"\nCom {num_neurons} neurônios, você pode classificar até {max_classes} classes")

    if len(set(y)) > max_classes:
        print(f"ERRO: Você tem {len(set(y))} classes mas só {max_classes} são suportados!")
        exit(1)

    # Criar e treinar
    model = MultiPerceptron(input_size=256, num_neurons=num_neurons, learning_rate=0.01)

    print("\nTreinando...\n")
    train_info = model.train(X, y, epochs=epochs)

    print(f"\n{'='*50}")
    print(f"Treinamento concluído!")
    print(f"{'='*50}")
    print(f"Neurônios: {train_info['neurons']}")
    print(f"Épocas executadas: {train_info['epochs_executed']}")
    print(f"Convergiu: {'Sim ✓' if train_info['convergence'] else 'Não ✗'}")
    print(f"\nMapeamento de labels:")
    for label, code in train_info['label_mapping'].items():
        print(f"  {label} -> {''.join(map(str, code))}")

    # Avaliar
    print(f"\n{'='*50}")
    print(f"Avaliação")
    print(f"{'='*50}")
    metrics = model.evaluate(X, y)
    print(f"Acurácia: {metrics['accuracy']:.2%}")
    print(f"Corretos: {metrics['correct']}/{metrics['total']}")

    # Salvar
    save = input("\nDeseja salvar o modelo? (s/n): ").strip().lower()
    if save == 's':
        filename = f"multi_perceptron_{num_neurons}n.json"
        model.save_model(filename)
