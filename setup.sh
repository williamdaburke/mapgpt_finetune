#!/bin/bash

# Create and activate virtual environment
python3 -m venv env
source env/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies from requirements.txt
pip install -r requirements.txt

# Download spaCy language models
python -m spacy download en_core_web_sm
python -m spacy download fr_core_news_sm
python -m spacy download es_core_news_sm
python -m spacy download ko_core_news_sm
python -m spacy download de_core_news_sm
python -m spacy download ja_core_news_sm

echo "Setup complete. To activate the virtual environment, run 'source env/bin/activate'"
