# -*- coding: utf-8 -*-
"""
finetune model
"""

import os
import re
import json
import pandas as pd
import torch
import spacy
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments
)

if os.getenv("WANDB_API_KEY") is not None:
    import wandb
    wandb.login(key=os.getenv("WANDB_API_KEY"))

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

def load_data(parquet_path='data.parquet'):
    print("Loading data from:", parquet_path)
    df_raw = pd.read_parquet(parquet_path, engine='pyarrow')
    print(f"Loaded {len(df_raw)} rows.")
    # Print count per label
    print("Count per label:")
    for label, count in df_raw['label'].value_counts().items():
        print(f"  {label}: {count}")
    # Print count per label
    print("Count per language:")
    for label, count in df_raw['language'].value_counts().items():
        print(f"  {label}: {count}")
    return df_raw

def combine_context(row):
    chat_history = ' [SEP] '.join(x['content'] for x in json.loads(row['chat_history']))
    return f"{chat_history} [SEP] {row['last_user_utterance']}"

def prepare_labels(df):
    print("Encoding labels...")
    label2id = {label: idx for idx, label in enumerate(df['label'].unique())}
    id2label = {v: k for k, v in label2id.items()}
    df['label_id'] = df['label'].map(label2id)
    return label2id, id2label

def load_spacy_models():
    print("Loading spaCy language models...")
    models = {
        'English': spacy.load('en_core_web_sm'),
        'French': spacy.load('fr_core_news_sm'),
        'Spanish': spacy.load('es_core_news_sm'),
        'Korean': spacy.load('ko_core_news_sm'),
        'German': spacy.load('de_core_news_sm'),
        'Japanese': spacy.load('ja_core_news_sm')
    }
    print("spaCy models loaded.")
    return models

def remove_numbers(text):
    return re.sub(r'\d+', '', text)

def remove_named_entities(text, lang_name, spacy_models):
    nlp = spacy_models.get(lang_name)
    if not nlp:
        return text
    doc = nlp(text)
    new_tokens = []
    for token in doc:
        if token.ent_type_:
            new_tokens.append(f"<{token.ent_type_}>")
        else:
            new_tokens.append(token.text)
    return " ".join(new_tokens)

def preprocess_multilingual(text, lang_name, spacy_models):
    text = remove_numbers(text)
    text = remove_named_entities(text, lang_name, spacy_models)
    return text

class IntentDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        return {
            'input_ids': torch.tensor(self.encodings['input_ids'][idx]),
            'attention_mask': torch.tensor(self.encodings['attention_mask'][idx]),
            'labels': torch.tensor(self.labels[idx])
        }

    def __len__(self):
        return len(self.labels)

def compute_metrics(eval_pred):
    preds = eval_pred.predictions.argmax(axis=1)
    labels = eval_pred.label_ids
    return {
        'accuracy': accuracy_score(labels, preds),
        'f1_weighted': f1_score(labels, preds, average="weighted")
    }

def main():
    print("Starting fine-tuning process...")

    df_raw = load_data()

    print("Combining chat history and last utterances...")
    df_raw['combined_text'] = df_raw.apply(combine_context, axis=1)

    label2id, id2label = prepare_labels(df_raw)

    spacy_models = load_spacy_models()

    print("Preprocessing text data...")
    df_only_utterance_raw = df_raw.copy()
    df_combined_clean = df_raw.copy()
    df_only_utterance_clean = df_raw.copy()

    df_raw['text_str'] = df_raw['combined_text'].astype(str)
    df_only_utterance_raw['text_str'] = df_only_utterance_raw['last_user_utterance'].astype(str)

    df_combined_clean['text_str'] = df_combined_clean.apply(
        lambda row: preprocess_multilingual(row['combined_text'], row['language'], spacy_models), axis=1
    )
    df_only_utterance_clean['text_str'] = df_only_utterance_clean.apply(
        lambda row: preprocess_multilingual(row['last_user_utterance'], row['language'], spacy_models), axis=1
    )

    df = pd.concat([df_raw, df_only_utterance_raw, df_combined_clean, df_only_utterance_clean], ignore_index=True)
    print(f"Total examples after augmentation: {len(df)}")

    print("Splitting train and validation data...")
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        df['text_str'], df['label_id'], test_size=0.2, stratify=df['label_id'], random_state=42
    )

    print("Tokenizing text...")
    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-multilingual-cased')
    train_encodings = tokenizer(list(train_texts), truncation=True, padding=True)
    val_encodings = tokenizer(list(val_texts), truncation=True, padding=True)

    print("Creating dataset objects...")
    train_dataset = IntentDataset(train_encodings, list(train_labels))
    val_dataset = IntentDataset(val_encodings, list(val_labels))

    print("Loading model...")
    model = DistilBertForSequenceClassification.from_pretrained(
        'distilbert-base-multilingual-cased',
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id
    )

    training_args = TrainingArguments(
        output_dir='./finetuning_run',
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=10,
        eval_strategy="epoch",
        logging_dir='./logs',
        logging_strategy="steps",
        logging_steps=10,
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_weighted",
        report_to=["wandb"] if os.getenv("WANDB_API_KEY") else []
    )

    print("Beginning training...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics
    )

    trainer.train()

    print("Evaluating model...")
    metrics = trainer.evaluate()
    print("Final evaluation metrics:", metrics)

    print("Saving model and tokenizer...")
    model.save_pretrained("./intent_classifier_distilbert")
    tokenizer.save_pretrained("./intent_classifier_distilbert")
    trainer.save_model("./intent_classifier_distilbert_model")

    print("Run finished")

if __name__ == "__main__":
    main()
