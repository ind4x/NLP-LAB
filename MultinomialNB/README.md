# Spam classification with Multinomial Naive Bayes

This laboratory implements a supervised natural language processing pipeline
to classify text communications as either legitimate (*ham*) or unsolicited
(*spam*). The implementation demonstrates end-to-end NLP engineering: raw text
sanitization, vocabulary tokenization, bag-of-words vectorization, and
probabilistic classification using the Multinomial Naive Bayes algorithm.

## Contents

- Implementation notebook: [main.ipynb](./main.ipynb) containing the complete
  preprocessing pipeline, model training, evaluation metrics, and sample
  inference tests.
- Dataset: [SpamCollection.txt](./SpamCollection.txt) containing 5,574 labeled
  SMS and email messages formatted in tab-separated values.

## Methodology

### 1. Exploratory data analysis and label encoding
The raw dataset consists of two columns: `Label` and `Message`. Inspection
confirms zero missing records. The categorical target strings (`ham` and
`spam`) are mapped to binary numeric targets ($0$ for ham, $1$ for spam) using
`LabelEncoder`.

### 2. Text preprocessing and cleaning
Raw text is normalized through custom preprocessing logic:
- Case normalization: all text strings are converted to lowercase.
- Character filtering: non-alphabetic characters are stripped using regular
  expressions (`re.sub('[^a-z]', ' ', text)`).
- Tokenization and stopword removal: words are split into tokens and filtered
  against NLTK's standard English stopword lexicon to eliminate non-informative
  syntactic noise.

### 3. Feature extraction (Bag of Words)
After splitting the data into training and testing partitions, text sequences
are mapped into high-dimensional numerical vectors using `CountVectorizer`.
This step builds an ngram vocabulary across the training corpus and transforms
each message into a sparse word-count occurrence vector.

### 4. Classifier training
The transformed feature space is fitted using Scikit-learn's `MultinomialNB`
estimator. The multinomial distribution models discrete word count frequencies,
applying Laplace smoothing to handle zero-frequency terms in unseen test
messages.

## Experimental results

The model was evaluated on a test holdout partition of 1,115 samples:

| Class | Precision | Recall | F1-Score | Support |
| :--- | :--- | :--- | :--- | :--- |
| **Ham (0)** | 0.99 | 1.00 | 1.00 | 970 |
| **Spam (1)** | 0.97 | 0.97 | 0.97 | 145 |
| **Overall Accuracy** | - | - | **0.9919** | 1,115 |

### Confusion matrix analysis

```text
               Predicted Ham    Predicted Spam
Actual Ham          966               4
Actual Spam           5             140
```

- True Negatives (correctly identified ham): 966
- False Positives (legitimate messages flagged as spam): 4
- False Negatives (spam messages missed): 5
- True Positives (correctly identified spam): 140

The low false-positive rate (0.41%) demonstrates that legitimate communications
are preserved with minimal misclassification risk.

## Tech stack

- Language: Python 3
- Data manipulation: Pandas, NumPy
- Natural language processing: NLTK, Regex
- Machine learning: Scikit-learn (`MultinomialNB`, `CountVectorizer`, `LabelEncoder`)

Authored by Youssef Fellah.
Developed for the Engineering Cycle at Mundiapolis University.
