import numpy as np
import sqlite3
import json
from typing import List, Tuple, Dict
from datetime import datetime


class MLP:
    """
    Multi-Layer Perceptron (Rede Neural com múltiplas camadas).

    Usa função de ativação sigmoid e algoritmo de backpropagation para treinamento.
    Arquitetura: camada de entrada -> camadas ocultas -> camada de saída
    """

    def __init__(self, layer_sizes: List[int], learning_rate: float = 0.1):
        """
        Inicializa a rede neural.

        Args:
            layer_sizes: Lista com o tamanho de cada camada
                        Ex: [256, 128, 64, 10] = entrada(256) -> oculta1(128) -> oculta2(64) -> saída(10)
            learning_rate: Taxa de aprendizado
        """
        self.layer_sizes = layer_sizes
        self.learning_rate = learning_rate
        self.num_layers = len(layer_sizes)

        # Inicializar pesos e biases para cada camada
        # weights[i] conecta a camada i com a camada i+1
        self.weights = []
        self.biases = []

        # Inicialização Xavier/Glorot para melhor convergência
        for i in range(self.num_layers - 1):
            # Xavier initialization
            limit = np.sqrt(6 / (layer_sizes[i] + layer_sizes[i+1]))
            w = np.random.uniform(-limit, limit, (layer_sizes[i], layer_sizes[i+1]))
            b = np.zeros((1, layer_sizes[i+1]))

            self.weights.append(w)
            self.biases.append(b)

        # Mapeamento de labels
        self.label_to_index = {}
        self.index_to_label = {}

    @staticmethod
    def sigmoid(z: np.ndarray) -> np.ndarray:
        """Função de ativação sigmoid."""
        # Clip para evitar overflow
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))

    @staticmethod
    def sigmoid_derivative(a: np.ndarray) -> np.ndarray:
        """Derivada da sigmoid: σ'(z) = σ(z) * (1 - σ(z))"""
        return a * (1 - a)

    def forward_propagation(self, X: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Propagação forward através da rede.

        Args:
            X: Entrada (batch_size, input_size) ou (input_size,)

        Returns:
            Tupla (activations, z_values)
            - activations: lista de ativações de cada camada
            - z_values: lista de valores pré-ativação (z = W*a + b)
        """
        # Garantir que X é 2D
        if X.ndim == 1:
            X = X.reshape(1, -1)

        activations = [X]  # a[0] = entrada
        z_values = []      # valores antes da ativação

        # Forward através de cada camada
        for i in range(self.num_layers - 1):
            z = np.dot(activations[i], self.weights[i]) + self.biases[i]
            a = self.sigmoid(z)

            z_values.append(z)
            activations.append(a)

        return activations, z_values

    def backward_propagation(self, X: np.ndarray, y: np.ndarray,
                            activations: List[np.ndarray],
                            z_values: List[np.ndarray]) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Backpropagation para calcular gradientes.

        Args:
            X: Entrada
            y: Saída esperada (one-hot encoded)
            activations: Ativações de forward propagation
            z_values: Valores z de forward propagation

        Returns:
            Tupla (weight_gradients, bias_gradients)
        """
        m = X.shape[0]  # batch size

        # Listas para armazenar gradientes
        weight_gradients = [np.zeros_like(w) for w in self.weights]
        bias_gradients = [np.zeros_like(b) for b in self.biases]

        # Erro da última camada (saída)
        # δ^L = (a^L - y) ⊙ σ'(z^L)
        delta = activations[-1] - y

        # Backpropagate através das camadas
        for l in reversed(range(self.num_layers - 1)):
            # Gradientes para esta camada
            weight_gradients[l] = np.dot(activations[l].T, delta) / m
            bias_gradients[l] = np.sum(delta, axis=0, keepdims=True) / m

            # Se não é a primeira camada, propagar erro para trás
            if l > 0:
                # δ^l = (δ^(l+1) * W^(l+1)^T) ⊙ σ'(z^l)
                delta = np.dot(delta, self.weights[l].T) * self.sigmoid_derivative(activations[l])

        return weight_gradients, bias_gradients

    def update_weights(self, weight_gradients: List[np.ndarray],
                      bias_gradients: List[np.ndarray]):
        """Atualiza pesos usando gradiente descendente."""
        for i in range(len(self.weights)):
            self.weights[i] -= self.learning_rate * weight_gradients[i]
            self.biases[i] -= self.learning_rate * bias_gradients[i]

    def encode_labels(self, labels: List[str]) -> Dict[str, int]:
        """
        Cria mapeamento de labels para índices.

        Args:
            labels: Lista de labels (strings)

        Returns:
            Dicionário label -> índice
        """
        unique_labels = sorted(set(labels))

        if len(unique_labels) != self.layer_sizes[-1]:
            raise ValueError(
                f"Número de classes ({len(unique_labels)}) deve ser igual ao "
                f"tamanho da camada de saída ({self.layer_sizes[-1]})"
            )

        for i, label in enumerate(unique_labels):
            self.label_to_index[label] = i
            self.index_to_label[i] = label

        return self.label_to_index

    def labels_to_one_hot(self, labels: List[str]) -> np.ndarray:
        """Converte labels para one-hot encoding."""
        num_classes = self.layer_sizes[-1]
        one_hot = np.zeros((len(labels), num_classes))

        for i, label in enumerate(labels):
            index = self.label_to_index[label]
            one_hot[i, index] = 1

        return one_hot

    def compute_loss(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """
        Calcula cross-entropy loss.

        L = -1/m * Σ[y * log(ŷ) + (1-y) * log(1-ŷ)]
        """
        m = y_true.shape[0]
        # Adicionar epsilon para evitar log(0)
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

        loss = -np.sum(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)) / m
        return loss

    def train(self, X_train: np.ndarray, y_train: List[str],
              epochs: int = 1000, batch_size: int = 32,
              validation_split: float = 0.0) -> Dict:
        """
        Treina a rede neural.

        Args:
            X_train: Padrões de entrada (N × input_size)
            y_train: Labels (lista de strings)
            epochs: Número de épocas
            batch_size: Tamanho do batch para mini-batch gradient descent
            validation_split: Fração dos dados para validação (0.0 a 1.0)

        Returns:
            Dicionário com informações do treinamento
        """
        # Criar codificação de labels
        self.encode_labels(y_train)
        y_encoded = self.labels_to_one_hot(y_train)

        # Split treino/validação se necessário
        if validation_split > 0:
            split_idx = int(len(X_train) * (1 - validation_split))
            X_val = X_train[split_idx:]
            y_val = y_encoded[split_idx:]
            X_train = X_train[:split_idx]
            y_encoded = y_encoded[:split_idx]
        else:
            X_val = None
            y_val = None

        num_samples = X_train.shape[0]
        history = {
            'train_loss': [],
            'train_accuracy': [],
            'val_loss': [],
            'val_accuracy': []
        }

        print(f"Iniciando treinamento: {epochs} épocas, batch size {batch_size}")
        print(f"Amostras de treino: {num_samples}")
        if X_val is not None:
            print(f"Amostras de validação: {X_val.shape[0]}")
        print(f"Arquitetura: {' -> '.join(map(str, self.layer_sizes))}")
        print()

        for epoch in range(epochs):
            # Embaralhar dados
            indices = np.random.permutation(num_samples)
            X_shuffled = X_train[indices]
            y_shuffled = y_encoded[indices]

            # Mini-batch gradient descent
            for i in range(0, num_samples, batch_size):
                X_batch = X_shuffled[i:i+batch_size]
                y_batch = y_shuffled[i:i+batch_size]

                # Forward propagation
                activations, z_values = self.forward_propagation(X_batch)

                # Backward propagation
                weight_grads, bias_grads = self.backward_propagation(
                    X_batch, y_batch, activations, z_values
                )

                # Update weights
                self.update_weights(weight_grads, bias_grads)

            # Calcular métricas a cada época
            if (epoch + 1) % 10 == 0 or epoch == 0:
                # Treino
                train_activations, _ = self.forward_propagation(X_train)
                train_pred = train_activations[-1]
                train_loss = self.compute_loss(train_pred, y_encoded)
                train_accuracy = np.mean(np.argmax(train_pred, axis=1) == np.argmax(y_encoded, axis=1))

                history['train_loss'].append(train_loss)
                history['train_accuracy'].append(train_accuracy)

                print(f"Época {epoch+1}/{epochs} - "
                      f"Loss: {train_loss:.4f} - "
                      f"Acc: {train_accuracy:.4f}", end="")

                # Validação
                if X_val is not None:
                    val_activations, _ = self.forward_propagation(X_val)
                    val_pred = val_activations[-1]
                    val_loss = self.compute_loss(val_pred, y_val)
                    val_accuracy = np.mean(np.argmax(val_pred, axis=1) == np.argmax(y_val, axis=1))

                    history['val_loss'].append(val_loss)
                    history['val_accuracy'].append(val_accuracy)

                    print(f" - Val Loss: {val_loss:.4f} - Val Acc: {val_accuracy:.4f}")
                else:
                    print()

        # Métricas finais
        final_activations, _ = self.forward_propagation(X_train)
        final_pred = final_activations[-1]
        final_loss = self.compute_loss(final_pred, y_encoded)
        final_accuracy = np.mean(np.argmax(final_pred, axis=1) == np.argmax(y_encoded, axis=1))

        return {
            'epochs': epochs,
            'final_loss': final_loss,
            'final_accuracy': final_accuracy,
            'architecture': self.layer_sizes,
            'label_mapping': self.label_to_index,
            'history': history
        }

    def predict(self, X: np.ndarray) -> str:
        """
        Faz predição para uma entrada.

        Args:
            X: Padrão de entrada

        Returns:
            Label predito (string)
        """
        activations, _ = self.forward_propagation(X)
        output = activations[-1]

        predicted_index = np.argmax(output)
        return self.index_to_label.get(predicted_index, "desconhecido")

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Retorna probabilidades de cada classe.

        Args:
            X: Padrão de entrada

        Returns:
            Array com probabilidades de cada classe
        """
        activations, _ = self.forward_propagation(X)
        return activations[-1]

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
        """Salva o modelo completo."""
        model_data = {
            'layer_sizes': self.layer_sizes,
            'learning_rate': self.learning_rate,
            'label_to_index': self.label_to_index,
            'index_to_label': {str(k): v for k, v in self.index_to_label.items()},
            'weights': [w.tolist() for w in self.weights],
            'biases': [b.tolist() for b in self.biases]
        }

        with open(filepath, 'w') as f:
            json.dump(model_data, f)

        print(f"Modelo salvo em {filepath}")

    def load_model(self, filepath: str):
        """Carrega o modelo completo."""
        with open(filepath, 'r') as f:
            model_data = json.load(f)

        self.layer_sizes = model_data['layer_sizes']
        self.learning_rate = model_data['learning_rate']
        self.num_layers = len(self.layer_sizes)
        self.label_to_index = model_data['label_to_index']
        self.index_to_label = {int(k): v for k, v in model_data['index_to_label'].items()}

        # Restaurar pesos e biases
        self.weights = [np.array(w) for w in model_data['weights']]
        self.biases = [np.array(b) for b in model_data['biases']]

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
    """Exemplo de uso do MLP."""
    print("=== Multi-Layer Perceptron (MLP) ===\n")

    # Carregar dados
    X, y = load_patterns_from_db()

    print(f"Dados carregados:")
    print(f"- Total de exemplos: {len(X)}")
    print(f"- Classes únicas: {set(y)}")
    print(f"- Número de classes: {len(set(y))}\n")

    num_classes = len(set(y))

    # Configurar arquitetura
    print("Configure a arquitetura da rede:")
    print("Exemplo: 256,128,64,10 = entrada(256) -> oculta1(128) -> oculta2(64) -> saída(10)")
    arch_input = input(f"Arquitetura (entrada fixa em 256, saída em {num_classes}): ")

    if arch_input.strip():
        hidden_layers = [int(x.strip()) for x in arch_input.split(',')]
        layer_sizes = [256] + hidden_layers + [num_classes]
    else:
        # Arquitetura padrão
        layer_sizes = [256, 128, num_classes]

    epochs = int(input("Quantas épocas? (padrão 500): ") or "500")
    batch_size = int(input("Batch size? (padrão 32): ") or "32")
    learning_rate = float(input("Learning rate? (padrão 0.1): ") or "0.1")

    print(f"\nArquitetura: {' -> '.join(map(str, layer_sizes))}")

    # Criar e treinar
    model = MLP(layer_sizes=layer_sizes, learning_rate=learning_rate)

    print("\nTreinando...\n")
    train_info = model.train(X, y, epochs=epochs, batch_size=batch_size)

    print(f"\n{'='*50}")
    print(f"Treinamento concluído!")
    print(f"{'='*50}")
    print(f"Arquitetura: {' -> '.join(map(str, train_info['architecture']))}")
    print(f"Épocas: {train_info['epochs']}")
    print(f"Loss final: {train_info['final_loss']:.4f}")
    print(f"Acurácia final: {train_info['final_accuracy']:.4f}")

    # Avaliar
    print(f"\n{'='*50}")
    print(f"Avaliação no conjunto de treino")
    print(f"{'='*50}")
    metrics = model.evaluate(X, y)
    print(f"Acurácia: {metrics['accuracy']:.2%}")
    print(f"Corretos: {metrics['correct']}/{metrics['total']}")

    # Salvar
    save = input("\nDeseja salvar o modelo? (s/n): ").strip().lower()
    if save == 's':
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"mlp_{len(layer_sizes)}layers_{timestamp}.json"
        model.save_model(filename)
