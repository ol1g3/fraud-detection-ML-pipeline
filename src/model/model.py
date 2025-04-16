import numpy as np
from scipy.stats import truncnorm
from typing import Tuple, Callable, List
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
)


def truncated_normal(
    mean: float = 0, sd: float = 1, low: float = 0, upp: float = 10
) -> Callable:
    """
    Create a truncated normal distribution.

    Args:
        mean: Mean of the distribution
        sd: Standard deviation
        low: Lower bound
        upp: Upper bound

    Returns:
        Truncated normal distribution
    """
    return truncnorm((low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)


class FraudDetectionModel:
    """
    A neural network model for fraud detection with two hidden layers.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [32, 16],
        learning_rate: float = 0.001,
        regularization: float = 0.2,
        activation: str = "relu",
        dropout: float = 0.0,
        batch_size: int = 32,
    ):
        """
        Initialize the neural network model.

        Args:
            input_dim: Number of input features
            hidden_dims: List of hidden layer dimensions [first_layer, second_layer]
            learning_rate: Learning rate for gradient descent
            regularization: Regularization strength (L2)
            activation: Activation function for hidden layers ('relu' or 'sigmoid')
            dropout: Dropout rate (0-1)
            batch_size: Mini-batch size for training
        """
        self.input_dim = input_dim
        self.output_dim = 1

        if len(hidden_dims) != 2:
            raise ValueError("Expected exactly 2 hidden dimensions")
        self.hidden_dims = hidden_dims

        self.learning_rate = learning_rate
        self.regularization = regularization
        self.activation = activation
        self.dropout_rate = dropout
        self.batch_size = batch_size

        self._create_weight_matrices()

    def _create_weight_matrices(self) -> None:
        X = truncated_normal(mean=0, sd=1, low=-0.5, upp=0.5)

        self.w1 = X.rvs(self.hidden_dims[0] * self.input_dim).reshape(
            (self.hidden_dims[0], self.input_dim)
        )

        self.w2 = X.rvs(self.hidden_dims[1] * self.hidden_dims[0]).reshape(
            (self.hidden_dims[1], self.hidden_dims[0])
        )

        self.w3 = X.rvs(self.output_dim * self.hidden_dims[1]).reshape(
            (self.output_dim, self.hidden_dims[1])
        )

    def _activate(self, x: np.ndarray) -> np.ndarray:
        if self.activation == "sigmoid":
            return 1 / (1 + np.exp(-x))
        elif self.activation == "relu":
            return np.maximum(0, x)
        else:
            raise ValueError(f"Unknown activation function: {self.activation}")

    def _activate_derivative(self, x: np.ndarray) -> np.ndarray:
        if self.activation == "sigmoid":
            sig = 1 / (1 + np.exp(-x))
            return sig * (1 - sig)
        elif self.activation == "relu":
            return np.where(x > 0, 1, 0)
        else:
            raise ValueError(f"Unknown activation function: {self.activation}")

    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-x))

    def _apply_dropout(self, x: np.ndarray) -> np.ndarray:
        if self.dropout_rate > 0:
            mask = np.random.rand(*x.shape) > self.dropout_rate
            return x * mask / (1.0 - self.dropout_rate)
        return x

    def _forward(self, x: np.ndarray, training: bool = False) -> Tuple:
        """
        Forward pass through the network.

        Args:
            x: Input data
            training: Whether to apply dropout (for training)

        Returns:
            Tuple of activations and inputs at each layer
        """
        z1 = np.dot(self.w1, x)
        a1 = self._activate(z1)
        if training and self.dropout_rate > 0:
            a1 = self._apply_dropout(a1)

        z2 = np.dot(self.w2, a1)
        a2 = self._activate(z2)
        if training and self.dropout_rate > 0:
            a2 = self._apply_dropout(a2)

        z3 = np.dot(self.w3, a2)
        a3 = self._sigmoid(z3)

        return z1, a1, z2, a2, z3, a3

    def train_step(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        Perform a single training step (forward and backward pass) for a batch.

        Args:
            x: Input batch features (batch_size, input_dim)
            y: Input batch labels (batch_size,)

        Returns:
            Loss for the batch
        """
        x = np.array(x, ndmin=2).T
        y = np.array(y).reshape(1, -1)
        batch_size = x.shape[1]

        # Foward pass
        z1 = np.dot(self.w1, x)
        a1 = self._activate(z1)
        if self.dropout_rate > 0:
            a1 = self._apply_dropout(a1)

        z2 = np.dot(self.w2, a1)
        a2 = self._activate(z2)
        if self.dropout_rate > 0:
            a2 = self._apply_dropout(a2)

        z3 = np.dot(self.w3, a2)
        a3 = self._sigmoid(z3)

        # Compute Loss (Binary Cross-Entropy)
        epsilon = 1e-7
        loss = -np.mean(y * np.log(a3 + epsilon) + (1 - y) * np.log(1 - a3 + epsilon))

        # Backward pass
        delta3 = a3 - y

        delta2 = np.dot(self.w3.T, delta3) * self._activate_derivative(z2)

        delta1 = np.dot(self.w2.T, delta2) * self._activate_derivative(z1)

        dw3 = np.dot(delta3, a2.T) / batch_size + self.regularization * self.w3
        dw2 = np.dot(delta2, a1.T) / batch_size + self.regularization * self.w2
        dw1 = np.dot(delta1, x.T) / batch_size + self.regularization * self.w1

        # Update weights
        self.w3 -= self.learning_rate * dw3
        self.w2 -= self.learning_rate * dw2
        self.w1 -= self.learning_rate * dw1

        return loss

    def train(self, data: np.ndarray, labels: np.ndarray) -> float:
        """
        Train the model on a dataset using mini-batches.

        Args:
            data: Training features
            labels: Training targets

        Returns:
            Average loss
        """
        total_loss = 0
        n_batches = 0

        for i in range(0, len(data), self.batch_size):
            batch_x = data[i : i + self.batch_size]
            batch_y = labels[i : i + self.batch_size]

            if len(batch_x) > 0:
                batch_loss = self.train_step(batch_x, batch_y)
                total_loss += batch_loss
                n_batches += 1

        return total_loss / max(n_batches, 1)

    def predict(self, x: np.ndarray) -> float:
        """
        Make a prediction for an input.

        Args:
            x: Input features

        Returns:
            Probability prediction (0-1)
        """
        if x.ndim != 1:
            raise ValueError("Input must be a 1D array")
        x = np.array(x, ndmin=2).T
        _, _, _, _, _, output = self._forward(x, training=False)
        return output[0][0]

    def evaluate(self, data: np.ndarray, labels: np.ndarray) -> dict:
        """
        Evaluate model on a dataset.

        Args:
            data: Test features
            labels: Test targets

        Returns:
            Dictionary with accuracy and other metrics
        """
        correct = 0
        total = len(data)
        predictions = []

        for i in range(total):
            pred_prob = self.predict(data[i])
            pred_class = np.round(pred_prob)

            predictions.append(pred_class)

            if pred_class == labels[i]:
                correct += 1

        accuracy = correct / total
        return {
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
            "predictions": predictions,
        }

    def find_optimal_threshold(
        self, y_true: np.ndarray, y_scores: np.ndarray, optimize_for: str = "f1"
    ) -> float:
        """
        Find the optimal threshold based on a specified metric.

        Args:
            y_true: True labels
            y_scores: Predicted probabilities
            optimize_for: Metric to optimize ('f1', 'precision', 'recall')

        Returns:
            Optimal threshold value
        """
        thresholds = np.linspace(0.01, 0.99, 99)
        scores = []

        for threshold in thresholds:
            y_pred = (y_scores >= threshold).astype(int)

            if optimize_for == "precision":
                score = precision_score(y_true, y_pred, zero_division=0)
            elif optimize_for == "recall":
                score = recall_score(y_true, y_pred, zero_division=0)
            else:
                score = f1_score(y_true, y_pred, zero_division=0)

            scores.append(score)

        optimal_idx = np.argmax(scores)
        optimal_threshold = thresholds[optimal_idx]

        return optimal_threshold
