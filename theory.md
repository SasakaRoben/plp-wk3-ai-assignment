# Part 1: Theoretical Understanding (40%)

---

## 1. Short Answer Questions

### a) Explain the primary differences between TensorFlow and PyTorch. When would you choose one over the other?

**TensorFlow** and **PyTorch** are both popular deep learning frameworks, but they differ mainly in **execution style, usability, and ecosystem**:

| Aspect | TensorFlow | PyTorch |
|---------|-------------|----------|
| **Execution Mode** | Uses *static computation graphs* (via `tf.Graph`), though eager execution is now supported. | Uses *dynamic computation graphs*, allowing operations to run immediately (define-by-run). |
| **Ease of Debugging** | Harder to debug due to static graphs (though TensorFlow 2 made it easier). | Easier to debug using standard Python tools because of its dynamic nature. |
| **Deployment** | Excellent for production with TensorFlow Serving, TensorFlow Lite, and TensorFlow.js. | Deployment improving through TorchServe and ONNX but historically less mature. |
| **Performance** | Optimized for large-scale distributed training. | Very efficient for research prototyping and smaller-scale training. |
| **Ecosystem** | Larger ecosystem (e.g., Keras, TF Hub, TF Extended). | More Pythonic and flexible for experimentation. |

**Choose TensorFlow** when you need:
- Large-scale **production deployment**, **mobile deployment**, or **cross-platform** support.
- An integrated ecosystem (e.g., TensorFlow Extended for ML pipelines).

**Choose PyTorch** when you need:
- **Research and experimentation** with flexible, intuitive model building.
- Easier **debugging** and **Pythonic syntax** for rapid prototyping.

---

### b) Describe two use cases for Jupyter Notebooks in AI development.

1. **Model Prototyping and Experimentation**  
   Jupyter allows data scientists to iteratively test and visualize model performance, modify parameters, and instantly see results without running a full script.

2. **Educational Demonstrations and Documentation**  
   It’s ideal for teaching AI concepts or presenting models since code, visualizations, and explanations can all be embedded in a single interactive notebook.

---

### c) How does spaCy enhance NLP tasks compared to basic Python string operations?

While basic Python string operations can only perform simple manipulations (like splitting, searching, or replacing text), **spaCy** provides **advanced linguistic analysis** through pre-trained models and NLP pipelines.

| Feature | Python String Ops | spaCy |
|----------|------------------|--------|
| **Tokenization** | Manual splitting (e.g., `str.split()`) | Advanced tokenization handling punctuation, contractions, and special cases. |
| **Part-of-Speech Tagging** | Not available | Identifies grammatical roles (noun, verb, etc.). |
| **Named Entity Recognition** | Not available | Detects entities like names, dates, and locations. |
| **Dependency Parsing** | Not available | Analyzes grammatical structure and relationships. |

**In short:** spaCy transforms raw text into *structured linguistic data*, enabling downstream tasks like sentiment analysis, text classification, and chatbots.

---

## 2. Comparative Analysis

| Criteria | **Scikit-learn** | **TensorFlow** |
|-----------|------------------|----------------|
| **Target Applications** | Designed for **classical machine learning** (e.g., regression, SVM, decision trees, clustering). | Focused on **deep learning** (e.g., neural networks, CNNs, RNNs). |
| **Ease of Use for Beginners** | Very beginner-friendly; simple API (`fit()`, `predict()`), minimal setup required. | Steeper learning curve; requires understanding of tensors, layers, and computational graphs. |
| **Community Support** | Large and mature community for traditional ML; extensive documentation. | Extremely active community in AI and DL research; rich ecosystem (Keras, TF Hub, etc.). |

**Summary:**
- Use **Scikit-learn** for smaller datasets and traditional ML algorithms.  
- Use **TensorFlow** for complex deep learning models that benefit from GPU acceleration and large-scale deployment.

---
