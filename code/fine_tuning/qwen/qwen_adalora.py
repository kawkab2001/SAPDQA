import json
import torch
import re
import nltk
import numpy as np
import matplotlib.pyplot as plt
from datasets import Dataset
from transformers import (
    AutoTokenizer, AutoModelForQuestionAnswering,
    Trainer, TrainingArguments
)
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import meteor_score
from rouge import Rouge
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from peft import get_peft_model, AdaLoraConfig

# Download NLTK resources
nltk.download('punkt')
nltk.download('wordnet')

# Step 1: Load dataset
with open('complete_finall.json', 'r', encoding='latin1') as f:
    dataset = json.load(f)

# Step 2: Load model and tokenizer
model_name = "unsloth/Qwen2-0.5B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
base_model = AutoModelForQuestionAnswering.from_pretrained(model_name)

# Optionally, compute total_steps from your dataset size, batch size, and epochs.
# For example, if there are 1000 training samples, batch size=4, and 5 epochs, then:
# total_steps = (1000 // 4) * 5
# You can adapt this as needed.
total_steps = 1250

# Step 3: Apply AdaLoRA with valid task type "QUESTION_ANS" and total_step specified
adalora_config = AdaLoraConfig(
    task_type="QUESTION_ANS",
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],  # Update with the correct module names.
    lora_dropout=0.1,
    bias="none",
    total_step=total_steps
)






model = get_peft_model(base_model, adalora_config)

# Step 4: Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Step 5: Preprocess the dataset
def preprocess_data(dataset):
    processed_data = []
    for article in dataset["data"]:
        for paragraph in article["paragraphs"]:
            context = paragraph["context"]
            for qa in paragraph["qas"]:
                question = qa["question"]
                answer = qa["answers"][0]  # Assume one answer per question
                start_idx = answer["answer_start"]
                end_idx = start_idx + len(answer["text"])

                inputs = tokenizer(
                    question,
                    context,
                    max_length=512,
                    truncation=True,
                    padding="max_length",
                    return_offsets_mapping=True
                )
                offset_mapping = inputs.pop("offset_mapping")
                start_token = next((i for i, (s, e) in enumerate(offset_mapping) if s <= start_idx < e), None)
                end_token = next((i for i, (s, e) in reversed(list(enumerate(offset_mapping))) if start_idx <= e < end_idx), None)

                if start_token is not None and end_token is not None:
                    processed_data.append({
                        "context": context,
                        "question": question,
                        "answer_start": start_idx,
                        "answer_end": end_idx,
                        "start_token": start_token,
                        "end_token": end_token
                    })
    return processed_data

processed_dataset = preprocess_data(dataset)
hf_dataset = Dataset.from_list(processed_dataset)

# Step 6: Tokenize the dataset
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
    start_positions, end_positions = [], []

    for i, offsets in enumerate(offset_mapping):
        start_char = examples["answer_start"][i]
        end_char = examples["answer_end"][i]

        start_token = end_token = None
        for idx, (start, end) in enumerate(offsets):
            if start <= start_char < end:
                start_token = idx
            if start <= end_char <= end:
                end_token = idx
                break

        start_positions.append(start_token if start_token is not None else -1)
        end_positions.append(end_token if end_token is not None else -1)

    inputs["start_positions"] = start_positions
    inputs["end_positions"] = end_positions
    return inputs

tokenized_dataset = hf_dataset.map(tokenize_function, batched=True)

# Step 7: Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=1e-3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=7,
    weight_decay=0.01,
    save_steps=500,
    save_total_limit=2,
    logging_dir="./logs",
    logging_steps=10,
    fp16=True
)

# Step 8: Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    eval_dataset=tokenized_dataset,
    tokenizer=tokenizer
)

# Step 9: Train the model
trainer.train()

# Step 10: Save the fine-tuned model
model.save_pretrained("./fine_tuned_qa_model")
tokenizer.save_pretrained("./fine_tuned_qa_model")

# Step 11: Inference function
def answer_question(question, context):
    inputs = tokenizer.encode_plus(
        question, context, return_tensors="pt", max_length=512, truncation=True
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    start = torch.argmax(outputs.start_logits)
    end = torch.argmax(outputs.end_logits) + 1
    answer = tokenizer.convert_tokens_to_string(
        tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][start:end])
    )
    return answer

# Step 12: Define Evaluation Metrics Functions
rouge = Rouge()
sentence_model = SentenceTransformer('all-MiniLM-L6-v2')

def compute_f1(pred, true):
    p_tokens = tokenizer.tokenize(pred)
    t_tokens = tokenizer.tokenize(true)
    common = set(p_tokens) & set(t_tokens)
    if not common:
        return 0
    precision = len(common) / len(p_tokens)
    recall = len(common) / len(t_tokens)
    return 2 * (precision * recall) / (precision + recall)

def compute_bleu(pred, true):
    return sentence_bleu([tokenizer.tokenize(true)], tokenizer.tokenize(pred))

def compute_rouge(pred, true):
    return rouge.get_scores(pred, true)[0]['rouge-l']['f']

def compute_meteor(pred, true):
    return meteor_score([tokenizer.tokenize(true)], tokenizer.tokenize(pred))

def compute_confidence(outputs, start, end):
    s_probs = torch.softmax(outputs.start_logits, dim=-1)
    e_probs = torch.softmax(outputs.end_logits, dim=-1)
    return ((s_probs[0][start] + e_probs[0][end]) / 2).item()

def compute_similarity(pred, true):
    p_vec = sentence_model.encode(pred, convert_to_tensor=True).cpu().numpy()
    t_vec = sentence_model.encode(true, convert_to_tensor=True).cpu().numpy()
    return cosine_similarity([p_vec], [t_vec])[0][0]

def normalize_text(text):
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]", " ", text.lower())).strip()

def compute_accuracy(pred, true):
    return int(normalize_text(pred).split() == normalize_text(true).split())

# Step 13: Evaluate the model on a subset of the dataset
metrics = {k: [] for k in ["f1", "bleu", "rouge", "meteor", "confidence", "similarity", "accuracy"]}

for article in dataset["data"][:2]:  # Adjust subset size as needed
    for para in article["paragraphs"]:
        context = para["context"]
        for qa in para["qas"]:
            question = qa["question"]
            true_ans = qa["answers"][0]["text"]
            pred_ans = answer_question(question, context)

            inputs = tokenizer.encode_plus(
                question, context, return_tensors="pt", max_length=512, truncation=True,
                return_offsets_mapping=True
            ).to(device)
            offset = inputs.pop("offset_mapping")
            start_char = qa["answers"][0]["answer_start"]
            end_char = start_char + len(true_ans)
            start_token = end_token = None
            for idx, (s, e) in enumerate(offset[0]):
                if s <= start_char < e:
                    start_token = idx
                if s <= end_char <= e:
                    end_token = idx
                    break

            if start_token is None or end_token is None:
                continue
            with torch.no_grad():
                out = model(**inputs)

            metrics["confidence"].append(compute_confidence(out, start_token, end_token))
            metrics["f1"].append(compute_f1(pred_ans, true_ans))
            metrics["bleu"].append(compute_bleu(pred_ans, true_ans))
            metrics["rouge"].append(compute_rouge(pred_ans, true_ans))
            metrics["meteor"].append(compute_meteor(pred_ans, true_ans))
            metrics["similarity"].append(compute_similarity(pred_ans, true_ans))
            metrics["accuracy"].append(compute_accuracy(pred_ans, true_ans))

# Step 14: Print Average Metrics
for key in metrics:
    print(f"Average {key.upper()}: {sum(metrics[key]) / len(metrics[key]):.4f}")

# Step 15: Plot Training Loss
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
plt.savefig("/home/kawkab/training_loss_plot.png")
plt.show()
