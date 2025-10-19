#!/usr/bin/env python3
"""
MLP with Human-in-the-Loop (HITL) capabilities.

Features:
1. Feedback during prediction: Human corrects wrong predictions
2. Active Learning: System asks for feedback on uncertain cases
3. RLHF: Adjust weights based on human corrections
"""
import numpy as np
import json
from typing import List, Tuple, Dict, Optional
from mlp import MLP


class MLP_HITL(MLP):
    """
    Multi-Layer Perceptron with Human-in-the-Loop Learning.

    Extends the base MLP with:
    - Uncertainty estimation
    - Active learning queries
    - Online learning from human feedback
    - RLHF-style weight adjustments
    """

    def __init__(self, layer_sizes: List[int], learning_rate: float = 0.1):
        super().__init__(layer_sizes, learning_rate)

        # Feedback tracking
        self.feedback_history = []
        self.uncertainty_threshold = 0.3  # Ask for feedback if uncertainty > 30%

    def predict_with_uncertainty(self, X: np.ndarray) -> Tuple[str, float, np.ndarray]:
        """
        Faz predição e calcula incerteza.

        Args:
            X: Padrão de entrada

        Returns:
            Tupla (predicted_label, uncertainty, probabilities)
            - uncertainty: 0.0 (certo) a 1.0 (muito incerto)
        """
        activations, _ = self.forward_propagation(X)
        probabilities = activations[-1][0]  # Shape: (num_classes,)

        # Predição
        predicted_index = np.argmax(probabilities)
        predicted_label = self.index_to_label.get(predicted_index, "desconhecido")

        # Calcular incerteza usando entropia
        # Entropia alta = incerto, entropia baixa = confiante
        epsilon = 1e-15
        probs_safe = np.clip(probabilities, epsilon, 1 - epsilon)
        entropy = -np.sum(probs_safe * np.log(probs_safe))

        # Normalizar entropia para [0, 1]
        max_entropy = -np.log(1.0 / len(probabilities))
        uncertainty = entropy / max_entropy if max_entropy > 0 else 0.0

        return predicted_label, uncertainty, probabilities

    def needs_feedback(self, uncertainty: float) -> bool:
        """
        Determina se deve pedir feedback humano (Active Learning).

        Args:
            uncertainty: Valor de incerteza (0.0 a 1.0)

        Returns:
            True se deve pedir feedback
        """
        return uncertainty > self.uncertainty_threshold

    def learn_from_feedback(self, X: np.ndarray, correct_label: str,
                           learning_rate_feedback: float = 0.5) -> Dict:
        """
        Aprende com correção humana (RLHF).

        Faz um passo de gradient descent usando o feedback como ground truth.

        Args:
            X: Padrão de entrada
            correct_label: Label correto fornecido pelo humano
            learning_rate_feedback: LR específico para feedback (maior = aprende mais rápido)

        Returns:
            Dict com informações do ajuste
        """
        # Converter label correto para one-hot
        if correct_label not in self.label_to_index:
            return {
                'success': False,
                'error': f'Label desconhecido: {correct_label}'
            }

        correct_index = self.label_to_index[correct_label]
        y_correct = np.zeros((1, len(self.label_to_index)))
        y_correct[0, correct_index] = 1

        # Forward propagation
        if X.ndim == 1:
            X = X.reshape(1, -1)

        activations, z_values = self.forward_propagation(X)

        # Predição antes do ajuste
        pred_before = np.argmax(activations[-1])

        # Backpropagation
        weight_grads, bias_grads = self.backward_propagation(X, y_correct, activations, z_values)

        # Salvar learning rate original
        original_lr = self.learning_rate

        # Usar learning rate maior para feedback
        self.learning_rate = learning_rate_feedback
        self.update_weights(weight_grads, bias_grads)

        # Restaurar learning rate
        self.learning_rate = original_lr

        # Predição após o ajuste
        activations_after, _ = self.forward_propagation(X)
        pred_after = np.argmax(activations_after[-1])

        # Registrar feedback
        feedback_record = {
            'timestamp': str(np.datetime64('now')),
            'correct_label': correct_label,
            'prediction_before': self.index_to_label[pred_before],
            'prediction_after': self.index_to_label[pred_after],
            'improved': (pred_after == correct_index) and (pred_before != correct_index)
        }

        self.feedback_history.append(feedback_record)

        return {
            'success': True,
            'improved': feedback_record['improved'],
            'prediction_before': feedback_record['prediction_before'],
            'prediction_after': feedback_record['prediction_after'],
            'total_feedback_count': len(self.feedback_history)
        }

    def get_feedback_stats(self) -> Dict:
        """
        Retorna estatísticas sobre o feedback recebido.

        Returns:
            Dict com estatísticas
        """
        if len(self.feedback_history) == 0:
            return {
                'total_feedback': 0,
                'improvements': 0,
                'improvement_rate': 0.0
            }

        improvements = sum(1 for f in self.feedback_history if f['improved'])

        return {
            'total_feedback': len(self.feedback_history),
            'improvements': improvements,
            'improvement_rate': improvements / len(self.feedback_history),
            'recent_feedback': self.feedback_history[-5:]  # Últimos 5
        }

    def batch_learn_from_feedback(self, feedback_batch: List[Tuple[np.ndarray, str]],
                                  learning_rate_feedback: float = 0.3) -> Dict:
        """
        Aprende de múltiplos feedbacks em batch.

        Args:
            feedback_batch: Lista de (pattern, correct_label)
            learning_rate_feedback: Learning rate para o batch

        Returns:
            Dict com resultados
        """
        results = []

        for pattern, correct_label in feedback_batch:
            result = self.learn_from_feedback(pattern, correct_label, learning_rate_feedback)
            results.append(result)

        successful = sum(1 for r in results if r.get('success', False))
        improved = sum(1 for r in results if r.get('improved', False))

        return {
            'batch_size': len(feedback_batch),
            'successful': successful,
            'improved': improved,
            'improvement_rate': improved / len(feedback_batch) if len(feedback_batch) > 0 else 0.0
        }

    def save_model(self, filepath: str):
        """Salva modelo incluindo histórico de feedback."""
        # Converter feedback_history para tipos nativos Python
        clean_feedback = []
        for feedback in self.feedback_history:
            clean_feedback.append({
                'timestamp': str(feedback['timestamp']),
                'correct_label': str(feedback['correct_label']),
                'prediction_before': str(feedback['prediction_before']),
                'prediction_after': str(feedback['prediction_after']),
                'improved': bool(feedback['improved'])
            })

        model_data = {
            'layer_sizes': self.layer_sizes,
            'learning_rate': float(self.learning_rate),
            'label_to_index': self.label_to_index,
            'index_to_label': {str(k): v for k, v in self.index_to_label.items()},
            'weights': [w.tolist() for w in self.weights],
            'biases': [b.tolist() for b in self.biases],
            'feedback_history': clean_feedback,
            'uncertainty_threshold': float(self.uncertainty_threshold)
        }

        with open(filepath, 'w') as f:
            json.dump(model_data, f)

        print(f"Modelo HITL salvo em {filepath}")

    def load_model(self, filepath: str):
        """Carrega modelo incluindo histórico de feedback."""
        super().load_model(filepath)

        # Carregar dados específicos HITL
        with open(filepath, 'r') as f:
            model_data = json.load(f)

        self.feedback_history = model_data.get('feedback_history', [])
        self.uncertainty_threshold = model_data.get('uncertainty_threshold', 0.3)

        print(f"Modelo HITL carregado com {len(self.feedback_history)} feedbacks")


if __name__ == '__main__':
    """Teste do sistema HITL."""
    print("=== Teste MLP com Human-in-the-Loop ===\n")

    # Criar dados de teste
    X_test = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ])
    y_test = ['A', 'B', 'B', 'A']

    # Criar modelo
    model = MLP_HITL(layer_sizes=[2, 4, 2], learning_rate=0.3)

    # Treinar rapidamente
    print("Treinando modelo base...")
    model.train(X_test, y_test, epochs=500, batch_size=4)
    print()

    # Testar predição com incerteza
    print("Testando predições com incerteza:")
    for i, x in enumerate(X_test):
        pred, uncertainty, probs = model.predict_with_uncertainty(x)
        true_label = y_test[i]

        print(f"Input: {x}")
        print(f"  Predição: {pred} (Real: {true_label})")
        print(f"  Incerteza: {uncertainty:.2%}")
        print(f"  Precisa feedback? {model.needs_feedback(uncertainty)}")
        print()

        # Simular feedback se errou
        if pred != true_label:
            print(f"  → Aplicando feedback: correção para '{true_label}'")
            result = model.learn_from_feedback(x, true_label)
            print(f"  → Melhorou? {result['improved']}")
            print(f"  → Antes: {result['prediction_before']}, Depois: {result['prediction_after']}")
            print()

    # Estatísticas
    stats = model.get_feedback_stats()
    print("Estatísticas de Feedback:")
    print(f"  Total: {stats['total_feedback']}")
    print(f"  Melhorias: {stats['improvements']}")
    print(f"  Taxa de melhoria: {stats['improvement_rate']:.1%}")
