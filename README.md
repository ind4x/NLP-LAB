# Natural language processing laboratory portfolio

This repository contains practical laboratory implementations and academic
reference materials completed as part of the Natural Language Processing (NLP)
and Artificial Intelligence curriculum. The work focuses on text processing,
vector space modeling, probabilistic classification, and machine learning
foundations.

## Repository structure

- [MultinomialNB](./MultinomialNB): End-to-end SMS and email spam
  classification system using NLTK, bag-of-words tokenization via
  `CountVectorizer`, and a Multinomial Naive Bayes classifier.
- [NeuralNetwork](./NeuralNetwork): From-scratch implementation of a two-layer
  Multi-Layer Perceptron (MLP) for non-linear XOR classification with
  analytical derivation, backpropagation, and batch gradient descent.
- [resources](./resources): Coursework problem statements, reference
  exercises, and supplementary diagrams covering supervised learning and
  gradient descent optimization.

## Key implementations

### SMS and email spam classification
The text classification laboratory implements an NLP pipeline on the SMS
Spam Collection dataset (5,574 messages).

Key metrics achieved on the test partition:
- Overall accuracy: 99.19%
- Precision (Spam class): 0.97
- Recall (Spam class): 0.97
- F1-score (Spam class): 0.97

The workflow demonstrates text sanitization, stopword removal, vocabulary
vectorization, model training under conditional independence assumptions, and
evaluation through confusion matrices and classification reports.

### Multi-layer perceptron for non-linear classification
The neural network laboratory models the non-linearly separable XOR logical gate
from first principles in NumPy without high-level deep learning frameworks.

Key components:
- Explicit analytical derivation for forward and backward passes.
- Exact verification comparing manual step calculations against vectorized
  NumPy operations.
- Modular `TwoLayerNeuralNetwork` class supporting vectorized batch propagation
  and gradient descent parameter optimization.
- Complete convergence on the XOR dataset across 10,000 training epochs.

## Tech stack

- Language: Python 3
- Environment: Jupyter Notebook, VS Code
- Data manipulation: Pandas, NumPy
- Natural language processing: NLTK, Regex
- Machine learning: Scikit-learn (`CountVectorizer`, `MultinomialNB`, `LabelEncoder`)

Authored by Youssef Fellah.
Developed for the Engineering Cycle at Mundiapolis University.
