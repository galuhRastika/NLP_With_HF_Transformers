<h1 align="center"> NLP with Hugging Face Transformers </h1>
<p align="center"> This project is a practical guide to using Hugging Face's Transformers library, illustrating how to load models, tokenize inputs, and perform NLP tasks like sentiment analysis. The notebook explores the power of Transformer-based models for natural language understanding, leveraging the simplicity and versatility of the Hugging Face ecosystem.</p>

<div align="center">
   
<img src="https://img.shields.io/badge/python-3.8%2B-blue">
<img src="https://img.shields.io/badge/Hugging%20Face-transformers-orange">
<img src="https://colab.research.google.com/assets/colab-badge.svg">
<img src="https://img.shields.io/badge/Jupyter-notebook-orange?logo=jupyter">
<img src="https://img.shields.io/badge/Notebook-Jupyter-informational?logo=python&color=green">
</div>

---

## 📝 Contents
1. Introduction to Transformers
   - Learn the basics of Transformer models and the Hugging Face library.
2. Loading and Tokenizing Models
   - Step-by-step guide to load pre-trained models and tokenizers.
3. Encoding Text and Making Predictions
   - Encode text data and run predictions using the model.
4. Pipeline Utility for NLP Tasks
   - Using the pipeline API for quick and effective text classification.

## 🚀 Getting Started

### 1. Setup
To get started, clone this repository and install the required libraries.

```bash
git clone <repo-url>
cd NLP-with-Transformers
pip install transformers torch
```



### 2. Load a Model and Tokenizer
The Hugging Face library makes it incredibly easy to load state-of-the-art models. In this example, we use the distilbert-base-uncased model, which is efficient and performs well on many NLP tasks.
```bash
python
from transformers import AutoTokenizer, AutoModel

model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
```


### 3. Tokenize Text Inputs
Transform raw text into model-compatible formats using the tokenizer.
```bash
python
inputs = tokenizer("Hello, how are you?", return_tensors="pt")
```

### 4. Run the Model
Once tokenized, pass the inputs to the model to obtain predictions.
```bash
python
outputs = model(**inputs)
```

### 5. Using the Pipeline API
For tasks like text classification, Hugging Face's pipeline simplifies the process:
```bash
python
from transformers import pipeline

classifier = pipeline("sentiment-analysis")
result = classifier("I love using Hugging Face Transformers!")
print(result)  # [{"label": "POSITIVE", "score": 0.99}]
```

## ✨ Analysis

This notebook showcases the ease of integrating Transformer models into projects using the Hugging Face API. By loading the model and tokenizer, we observe the importance of tokenization in preparing text data for model inference. API pipelines are especially useful for rapid prototyping, as they abstract away the complexity of NLP tasks such as sentiment analysis. You can check a more complete analysis at :arrow_down:
https://github.com/galuhRastika/NLP_With_HF_Transformers/blob/main/NLP_with_Hugging_Face_Transformers1_Galuh_Rastika_P.ipynb


### 📌 Key Observations

- *Flexibility*: Hugging Face supports a range of models and tokenizers, making it suitable for various NLP tasks.
- *Efficiency*: By using models like DistilBERT, it's possible to achieve high performance with less computational cost.
- *Ease of Use*: The pipeline utility provides an intuitive API for beginners and a rapid prototyping tool for developers.

## 🎨 Visualizing Model Outputs

Future enhancements could include visualizations for embeddings and attention patterns to better understand model predictions.

```bash
