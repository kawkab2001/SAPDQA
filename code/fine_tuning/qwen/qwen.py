import json
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, Trainer, TrainingArguments
import torch
from datasets import Dataset
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from nltk.translate.bleu_score import sentence_bleu
from rouge import Rouge
from nltk.translate.meteor_score import meteor_score
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from sentence_transformers import SentenceTransformer  # For similarity computation
import nltk

# Download NLTK resources
nltk.download('punkt')
nltk.download('wordnet')

# Step 1: Load the dataset
with open('complete_finall.json', 'r', encoding='latin1') as f:
    dataset = json.load(f)

# Step 2: Load the model and tokenizer
model_name = "unsloth/Qwen2-0.5B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForQuestionAnswering.from_pretrained(model_name)

# Move the model to the correct device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Step 3: Preprocess the dataset
def preprocess_data(dataset):
    processed_data = []
    for article in dataset["data"]:
        for paragraph in article["paragraphs"]:
            context = paragraph["context"]
            for qa in paragraph["qas"]:
                question = qa["question"]
                answer = qa["answers"][0]  # Assuming only one answer per question
                start_idx = answer["answer_start"]
                end_idx = start_idx + len(answer["text"])

                # Tokenize and check if the answer span can be mapped
                inputs = tokenizer(
                    question,
                    context,
                    max_length=512,
                    truncation=True,
                    padding="max_length",
                    return_offsets_mapping=True
                )
                offset_mapping = inputs.pop("offset_mapping")
                start_token = next((idx for idx, (start, _) in enumerate(offset_mapping) if start <= start_idx < _), None)
                end_token = next((idx for idx, (_, end) in reversed(list(enumerate(offset_mapping))) if start_idx <= end < end_idx), None)

                # Only include examples with valid token mappings
                if start_token is not None and end_token is not None:
                    processed_data.append({
                        "context": context,
                        "question": question,
                        "answer_start": start_idx,
                        "answer_end": end_idx,
                        "start_token": start_token,
                        "end_token": end_token
                    })
                else:
                    print(f"Skipping example: Question={question}, Context={context[:50]}...")
    return processed_data

processed_dataset = preprocess_data(dataset)

# Convert to Hugging Face Dataset format
hf_dataset = Dataset.from_list(processed_dataset)

# Tokenize the dataset
def tokenize_function(examples):
    inputs = tokenizer(
        examples["question"],
        examples["context"],
        max_length=512,
        truncation=True,
        padding="max_length",
        return_offsets_mapping=True
    )
    offset_mapping = inputs.pop("offset_mapping")
    start_positions = []
    end_positions = []

    for i, offsets in enumerate(offset_mapping):
        start_char = examples["answer_start"][i]
        end_char = examples["answer_end"][i]

        # Map character-level indices to token-level indices
        start_token = None
        end_token = None
        for idx, (start, end) in enumerate(offsets):
            if start <= start_char < end:
                start_token = idx
            if start <= end_char <= end:
                end_token = idx
                break  # Stop once we find the end token

        # If no valid mapping exists, skip this example
        if start_token is None or end_token is None:
            start_positions.append(-1)
            end_positions.append(-1)
        else:
            start_positions.append(start_token)
            end_positions.append(end_token)

    inputs["start_positions"] = start_positions
    inputs["end_positions"] = end_positions
    return inputs

tokenized_dataset = hf_dataset.map(tokenize_function, batched=True)

# Step 4: Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=1e-3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=5,
    weight_decay=0.01,
    save_steps=500,
    save_total_limit=2,
    logging_dir="./logs",
    logging_steps=10,
    fp16=True ,
   


  
  
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    eval_dataset=tokenized_dataset,
    tokenizer=tokenizer
)

# Step 6: Fine-tune the model
trainer.train()

# Step 7: Save the fine-tuned model
model.save_pretrained("./fine_tuned_qa_model")
tokenizer.save_pretrained("./fine_tuned_qa_model")

# Step 8: Test the fine-tuned model
def answer_question(question, context):
    # Tokenize input
    inputs = tokenizer.encode_plus(
        question,
        context,
        return_tensors="pt",
        max_length=512,
        truncation=True
    )

    # Move inputs to the same device as the model
    inputs = {key: value.to(device) for key, value in inputs.items()}

    # Get model predictions
    with torch.no_grad():
        outputs = model(**inputs)

    # Extract answer start and end positions
    answer_start = torch.argmax(outputs.start_logits)
    answer_end = torch.argmax(outputs.end_logits) + 1

    # Decode the answer
    answer = tokenizer.convert_tokens_to_string(
        tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][answer_start:answer_end])
    )

    return answer

# Evaluation Metrics Functions
def compute_f1(predicted_answer, true_answer):
    pred_tokens = tokenizer.tokenize(predicted_answer)
    true_tokens = tokenizer.tokenize(true_answer)
    common_tokens = set(pred_tokens) & set(true_tokens)
    if not common_tokens:
        return 0
    precision = len(common_tokens) / len(pred_tokens)
    recall = len(common_tokens) / len(true_tokens)
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return f1

def compute_bleu(predicted_answer, true_answer):
    reference = [tokenizer.tokenize(true_answer)]
    candidate = tokenizer.tokenize(predicted_answer)
    bleu_score = sentence_bleu(reference, candidate)
    return bleu_score

rouge = Rouge()
def compute_rouge(predicted_answer, true_answer):
    scores = rouge.get_scores(predicted_answer, true_answer)[0]
    return scores['rouge-l']['f']  # Use ROUGE-L F1 score

def compute_meteor(predicted_answer, true_answer):
    reference = [tokenizer.tokenize(true_answer)]
    candidate = tokenizer.tokenize(predicted_answer)
    meteor = meteor_score(reference, candidate)
    return meteor

def compute_confidence(outputs, start_token, end_token):
    start_probs = torch.softmax(outputs.start_logits, dim=-1)
    end_probs = torch.softmax(outputs.end_logits, dim=-1)
    confidence = (start_probs[0][start_token] + end_probs[0][end_token]) / 2
    return confidence.item()

sentence_model = SentenceTransformer('all-MiniLM-L6-v2')  # For similarity computation
def compute_similarity(predicted_answer, true_answer):
    pred_embedding = sentence_model.encode(predicted_answer, convert_to_tensor=True).cpu().numpy()
    true_embedding = sentence_model.encode(true_answer, convert_to_tensor=True).cpu().numpy()
    similarity = cosine_similarity([pred_embedding], [true_embedding])[0][0]
    return similarity
import re

def normalize_text(text):
    """
    Normalize the text by removing extra spaces, punctuation, and converting to lowercase.
    """
    text = text.lower()  # Convert to lowercase
    text = re.sub(r"[^a-z0-9]", " ", text)  # Remove non-alphanumeric characters
    text = re.sub(r"\s+", " ", text).strip()  # Remove extra spaces
    return text
def compute_accuracy(predicted_answer, true_answer):
    """
    Compute accuracy based on normalized tokenized text.
    """
    # Normalize both answers
    pred_tokens = normalize_text(predicted_answer).split()
    true_tokens = normalize_text(true_answer).split()

    # Check if the tokenized answers match exactly
    return 1 if pred_tokens == true_tokens else 0

# Evaluate the model
metrics = {
    "f1": [],
    "bleu": [],
    "rouge": [],
    "meteor": [],
    "confidence": [],
    "similarity": [],
    "accuracy": []
}

for article in dataset["data"][:2]:  # Evaluate on a subset of data
    for paragraph in article["paragraphs"]:
        context = paragraph["context"]
        for qa in paragraph["qas"]:
            question = qa["question"]
            true_answer = qa["answers"][0]["text"]

            # Predict the answer
            predicted_answer = answer_question(question, context)

            # Get tokenized input and offsets
            inputs = tokenizer.encode_plus(
                question,
                context,
                return_tensors="pt",
                max_length=512,
                truncation=True,
                return_offsets_mapping=True
            ).to(device)
            offset_mapping = inputs.pop("offset_mapping")

            # Map character-level indices to token-level indices
            start_char = qa["answers"][0]["answer_start"]
            end_char = start_char + len(true_answer)
            start_token = None
            end_token = None
            for idx, (start, end) in enumerate(offset_mapping[0]):  # Batch size is 1
                if start <= start_char < end:
                    start_token = idx
                if start <= end_char <= end:
                    end_token = idx
                    break

            # Skip invalid examples
            if start_token is None or end_token is None:
                print(f"Skipping invalid example: Question={question}")
                continue

            # Compute confidence
            with torch.no_grad():
                outputs = model(**inputs)
            metrics["confidence"].append(compute_confidence(outputs, start_token, end_token))

            # Compute other metrics
            metrics["f1"].append(compute_f1(predicted_answer, true_answer))
            metrics["bleu"].append(compute_bleu(predicted_answer, true_answer))
            metrics["rouge"].append(compute_rouge(predicted_answer, true_answer))
            metrics["meteor"].append(compute_meteor(predicted_answer, true_answer))
            metrics["similarity"].append(compute_similarity(predicted_answer, true_answer))
            metrics["accuracy"].append(compute_accuracy(predicted_answer, true_answer))

# Print Average Metrics
print("Average F1:", sum(metrics["f1"]) / len(metrics["f1"]))
print("Average BLEU:", sum(metrics["bleu"]) / len(metrics["bleu"]))
print("Average ROUGE:", sum(metrics["rouge"]) / len(metrics["rouge"]))
print("Average METEOR:", sum(metrics["meteor"]) / len(metrics["meteor"]))
print("Average Confidence:", sum(metrics["confidence"]) / len(metrics["confidence"]))
print("Average Similarity:", sum(metrics["similarity"]) / len(metrics["similarity"]))
print("Average Accuracy:", sum(metrics["accuracy"]) / len(metrics["accuracy"]))


log_history = trainer.state.log_history
train_losses = [log["loss"] for log in log_history if "loss" in log]
steps = list(range(1, len(train_losses) + 1))

plt.figure(figsize=(10, 6))
plt.plot(steps, train_losses, label="Training Loss", color="blue")
plt.xlabel("Steps")
plt.ylabel("Loss")
plt.title("Training Loss Over Time")
plt.legend()
plt.grid(True)
plt.savefig("/home/kawkab/training_loss_plot.png")  # Save the plot
plt.show()