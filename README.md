# Multilingual Intent Classification with DistilBERT

This project fine-tunes a multilingual DistilBERT model for intent classification using chat-based conversational data. It preprocesses chat histories in multiple languages, cleans and encodes the data, and trains a transformer model to classify user intents.

---

## Features

- Handles multilingual data (English, French, Spanish, Korean, German, Japanese).
- Combines chat history and last user utterance into a single input sequence.
- Cleans text by removing numbers and replacing named entities using spaCy language models.
- Uses Hugging Face Transformers and PyTorch for model training.
- Implements custom dataset class for efficient data handling.
- Evaluates model with accuracy and weighted F1-score metrics.
- Saves fine-tuned model and tokenizer for later use.

---

## Requirements

- Python 3.7+
- Packages listed in `requirements.txt`
- spaCy language models:
  - en_core_web_sm
  - fr_core_news_sm
  - es_core_news_sm
  - ko_core_news_sm
  - de_core_news_sm
  - ja_core_news_sm

See [`setup.sh`](setup.sh) for an automated setup script.

---

## Setup

1. Clone this repository.
2. Run the setup script to create a virtual environment and install dependencies:

   ```bash
   ./setup.sh

## Run

1. To log run to Weights & Biases, set environment variable `WANDB_API_KEY`.
2. To run the script locally: `python main.py`
3. To run the script in a notebook use `run.ipynb`. This allows the ability to run in a free GPU environment such a Google Colab or Kaggle if GPUs are not available locally.
