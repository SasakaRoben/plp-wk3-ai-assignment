# Part 1: Theoretical Understanding (40%)
## 1. Short Answer Questions
1. Explain the primary differences between TensorFlow and PyTorch. When would you choose one over the other?
2. Describe two use cases for Jupyter Notebooks in AI development.
3. How does spaCy enhance NLP tasks compared to basic Python string operations?

## 2. Comparative Analysis
Compare Scikit-learn and TensorFlow in terms of:
1. Target applications (e.g., classical ML vs. deep learning).
2. Ease of use for beginners.
3. Community support.

# Part 2: Practical Implementation (50%)
## Task 1: Classical ML with Scikit-learn

**Dataset**: Iris Species Dataset

**Goal:**
1. Preprocess the data (handle missing values, encode labels).
2. Train a decision tree classifier to predict iris species.
3. Evaluate using accuracy, precision, and recall.

**Deliverable**: Python script/Jupyter notebook with comments explaining each step.

## Task 2: Deep Learning with TensorFlow/PyTorch

**Dataset**: MNIST Handwritten Digits

**Goal:**

1. Build a CNN model to classify handwritten digits.
2. Achieve >95% test accuracy.
3. Visualize the model’s predictions on 5 sample images.

**Deliverable:** Code with model architecture, training loop, and evaluation.

## Task 3: NLP with spaCy

**Text Data:** User reviews from Amazon Product Reviews.

**Goal:**
1. Perform named entity recognition (NER) to extract product names and brands.
2. Analyze sentiment (positive/negative) using a rule-based approach.
3. Deliverable: Code snippet and output showing extracted entities and sentiment.

# Part 3: Ethics & Optimization (10%)
## 1. Ethical Considerations
Identify potential biases in your MNIST or Amazon Reviews model. How could tools like TensorFlow Fairness Indicators or spaCy’s rule-based systems mitigate these biases?

## 2. Troubleshooting Challenge
**Buggy Code:** A provided TensorFlow script has errors (e.g., dimension mismatches, incorrect loss functions). Debug and fix the code.

# Bonus Task (Extra 10%)
Deploy Your Model: Use Streamlit or Flask to create a web interface for your MNIST classifier. Submit a screenshot and a live demo link.