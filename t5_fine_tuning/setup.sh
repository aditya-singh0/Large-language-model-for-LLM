#!/bin/bash

# Create project directory structure
mkdir -p t5_fine_tuning/backend/api
mkdir -p t5_fine_tuning/pipeline
mkdir -p t5_fine_tuning/infrastructure
mkdir -p t5_fine_tuning/docs
mkdir -p t5_fine_tuning/tests

# Create requirements.txt in backend
cat <<EOL > t5_fine_tuning/backend/requirements.txt
flask
transformers
torch
EOL

# Create Flask API app.py
cat <<EOL > t5_fine_tuning/backend/api/app.py
from flask import Flask, request, jsonify
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch

app = Flask(__name__)
model = T5ForConditionalGeneration.from_pretrained('./fine-tuned-model')
tokenizer = T5Tokenizer.from_pretrained('./fine-tuned-model')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    input_text = data['input']
    input_ids = tokenizer.encode(input_text, return_tensors='pt')
    outputs = model.generate(input_ids)
    prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
EOL

# Create preprocess_data.py
cat <<EOL > t5_fine_tuning/pipeline/preprocess_data.py
from datasets import load_dataset
from transformers import T5Tokenizer

def preprocess_data(tokenizer, dataset_name="squad", split="train"):
    dataset = load_dataset(dataset_name, split=split)
    
    def preprocess_function(examples):
        inputs = [f"question: {q} context: {c}" for q, c in zip(examples["question"], examples["context"])]
        targets = [a["text"][0] for a in examples["answers"]]
        model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding="max_length")
        labels = tokenizer(targets, max_length=128, truncation=True, padding="max_length")
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    tokenized_dataset = dataset.map(preprocess_function, batched=True)
    return tokenized_dataset

if __name__ == "__main__":
    tokenizer = T5Tokenizer.from_pretrained("t5-small")
    tokenized_dataset = preprocess_data(tokenizer)
    print("Dataset preprocessed successfully.")
EOL

# Create fine_tune.py
cat <<EOL > t5_fine_tuning/pipeline/fine_tune.py
from transformers import T5ForConditionalGeneration, T5Tokenizer, Trainer, TrainingArguments
from preprocess_data import preprocess_data

def fine_tune_model(model_name="t5-small", dataset_name="squad"):
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)

    tokenized_dataset = preprocess_data(tokenizer, dataset_name)

    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        learning_rate=3e-5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=3,
        weight_decay=0.01,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        eval_dataset=tokenized_dataset,
    )

    trainer.train()
    model.save_pretrained("./fine-tuned-model")
    tokenizer.save_pretrained("./fine-tuned-model")

if __name__ == "__main__":
    fine_tune_model()
    print("Model fine-tuned and saved successfully.")
EOL

# Create evaluate.py
cat <<EOL > t5_fine_tuning/pipeline/evaluate.py
from transformers import T5ForConditionalGeneration, T5Tokenizer
from preprocess_data import preprocess_data

def evaluate_model(model_path="./fine-tuned-model", dataset_name="squad"):
    tokenizer = T5Tokenizer.from_pretrained(model_path)
    model = T5ForConditionalGeneration.from_pretrained(model_path)

    tokenized_dataset = preprocess_data(tokenizer, dataset_name, split="validation")

    inputs = tokenized_dataset["input_ids"]
    labels = tokenized_dataset["labels"]

    outputs = model.generate(inputs)
    predictions = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    references = tokenizer.batch_decode(labels, skip_special_tokens=True)

    for i in range(5):
        print(f"Prediction: {predictions[i]}")
        print(f"Reference: {references[i]}")

if __name__ == "__main__":
    evaluate_model()
    print("Model evaluated successfully.")
EOL

# Create Dockerfile
cat <<EOL > t5_fine_tuning/infrastructure/Dockerfile
FROM python:3.8-slim

WORKDIR /app

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY . .

CMD ["python", "api/app.py"]
EOL

# Create docker-compose.yml
cat <<EOL > t5_fine_tuning/infrastructure/docker-compose.yml
version: '3.8'

services:
  backend:
    build: ../backend
    ports:
      - "5000:5000"
    environment:
      - FLASK_ENV=development
EOL

# Create README.md
cat <<EOL > t5_fine_tuning/README.md
# T5 Model Fine-Tuning and Deployment

## Overview
This project demonstrates how to fine-tune the T5 model from Google Research and deploy it using a Flask API.

## Project Structure
- \`backend/\`: Contains the backend API for model inference.
- \`pipeline/\`: Contains scripts for loading the model, preparing data, fine-tuning, and evaluation.
- \`infrastructure/\`: Contains Docker scripts for infrastructure management.
- \`docs/\`: Documentation for the project.
- \`tests/\`: Unit tests for the project.

## Getting Started
1. Clone the repository.
2. Set up the backend:
   \`\`\`bash
   cd backend
   python -m venv venv
   source venv/bin/activate  # On Windows use \`venv\Scripts\activate\`
   pip install -r requirements.txt
   python api/app.py
   \`\`\`
3. Fine-tune the model:
   \`\`\`bash
   cd pipeline
   python fine_tune.py
   \`\`\`
4. Set up Docker:
   \`\`\`bash
   cd ../infrastructure
   docker-compose up --build
   \`\`\`

## Requirements
- Python 3.x
- PyTorch
- Hugging Face Transformers
- Flask
- Docker
EOL

echo "Setup script executed successfully."