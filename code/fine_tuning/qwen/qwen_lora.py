from unsloth import FastLanguageModel
import torch
from datasets import Dataset
import json
from transformers import GenerationConfig
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported
import matplotlib.pyplot as plt
import pickle
import torch
from tqdm import tqdm
import re
from nltk.tokenize import word_tokenize
from nltk.translate.meteor_score import meteor_score
max_seq_length = 2048
dtype = None
load_in_4bit = True
from nltk.translate.bleu_score import sentence_bleu
from rouge import Rouge
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import numpy as np
from nltk.translate.bleu_score import SmoothingFunction
import nltk
nltk.download('punkt_tab')
nltk.download('wordnet')
# Load model
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Qwen2-0.5B",
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)

# Prepare LoRA model
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj", ],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
    use_rslora=False,
    loftq_config=None,
)


# Load your custom dataset with answer positions
def load_custom_dataset(json_path):
    with open(json_path) as f:
        data = json.load(f)

    processed_data = []
    for item in data["data"]:
        for para in item["paragraphs"]:
            context = para["context"]
            for qa in para["qas"]:
                question = qa["question"]
                # Get all answer information including positions
                answers = qa["answers"]
                processed_data.append({
                    "context": context,
                    "question": question,
                    "answers": answers  # Now includes text, answer_start, answer_end
                })

    return Dataset.from_list(processed_data)

# Function to generate answers after fine-tuning
def generate_answer(context, question, return_confidence=True):
    # Prepare the input
    input_text = f"Context: {context}\nQuestion: {question}\nAnswer:"

    # Tokenize the input
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=max_seq_length).to("cuda")

    # Generate output with return_dict_in_generate to get scores
    generation_config = GenerationConfig(
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.1,
        max_new_tokens=200,
        return_dict_in_generate=True,
        output_scores=True,
    )

    outputs = model.generate(**inputs, generation_config=generation_config)

    # Decode the answer
    answer = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
    answer = answer.split("Answer:")[-1].strip()

    if not return_confidence:
        return answer

    # Calculate confidence score (average probability of the generated tokens)
    # Get the generated token ids
    gen_sequences = outputs.sequences[:, inputs.input_ids.shape[-1]:]

    # Get the logits for each generated token
    logits = torch.stack(outputs.scores, dim=1)

    # Convert logits to probabilities
    probs = torch.softmax(logits, dim=-1)

    # Get the probability of the generated tokens
    gen_probs = torch.gather(probs, 2, gen_sequences[:, :, None]).squeeze(-1)

    # Calculate average log probability (confidence score)
    confidence = torch.mean(gen_probs).item()

    return answer, confidence

# Modified formatting function to use answer positions
def formatting_prompts_func(examples):
    texts = []
    for context, question, answers in zip(examples["context"], examples["question"], examples["answers"]):
        # Take the first answer (you can modify this if needed)
        answer = answers[0]
        answer_text = answer["text"]
        answer_start = answer["answer_start"]
        answer_end = answer["answer_end"]

        # Format with context, question, answer text, and positions
        text = (f"Context: {context}\n"
                f"Question: {question}\n"
                f"Answer: {answer_text}\n"
                f"Answer Start: {answer_start}\n"
                f"Answer End: {answer_end}" + tokenizer.eos_token)
        texts.append(text)
    return {"text": texts}


# Load dataset
dataset = load_custom_dataset("complete_copy.json")


dataset = dataset.map(formatting_prompts_func, batched=True)

# Split dataset into train and eval (80-20 split)
dataset = dataset.train_test_split(test_size=0.2)
train_dataset = dataset["train"]
eval_dataset = dataset["test"]

# Add lists to store training metrics
train_losses = []
f1_scores = []
meteor_scores=[]

sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
rouge = Rouge()


def evaluate2(
        data2,
        output_file="eval_results.pkl",
        resume_from=0,
        word_match_threshold=0.7,
        require_all_words=False,
        max_samples=None,
        clear_cache_every=10,
):
    """
    Robust evaluation with empty answer handling and checkpointing.
    """
    # Initialize results
    results = {
        'accuracy': 0,
        'avg_word_overlap': 0,
        'avg_f1': 0,
        'avg_meteor': 0,
        'avg_bleu': 0,
        'avg_rouge': 0,
        'avg_similarity': 0,
        'avg_confidence': 0,
        'total_predictions': 0,
        'correct_predictions': 0,
        'processed_samples': 0,
    }

    # Load previous results if resuming
    try:
        with open(output_file, "rb") as f:
            saved_results = pickle.load(f)
            results.update(saved_results)
            print(f"Resuming from sample {resume_from}")
    except (FileNotFoundError, EOFError):
        print("Starting fresh evaluation")

    # Metric lists
    f1_scores = []
    meteor_scores = []
    bleu_scores = []
    rouge_scores = []
    similarity_scores = []
    confidence_scores = []
    word_overlap_scores = []

    eval_data = data2["data"][resume_from:max_samples]

    for i, item in enumerate(tqdm(eval_data, desc="Evaluating"), start=resume_from):
        if "paragraphs" not in item:
            continue

        for paragraph in item["paragraphs"]:
            context = paragraph["context"]

            for qa in paragraph["qas"]:
                if not qa.get("answers"):
                    continue

                for answer in qa["answers"]:
                    expected_answer = answer["text"].strip().lower()
                    predicted_answer, confidence = generate_answer(context, qa["question"])
                    predicted_answer = predicted_answer.strip().lower()

                    # Skip empty predictions
                    if not predicted_answer or not expected_answer:
                        continue

                    # --- Metric Calculations with Safeguards ---
                    try:
                        # Word Overlap
                        expected_words = set(re.findall(r'\w+', expected_answer))
                        predicted_words = set(re.findall(r'\w+', predicted_answer))
                        common_words = expected_words & predicted_words
                        overlap = len(common_words) / len(expected_words) if expected_words else 0
                        word_overlap_scores.append(overlap)

                        # Correctness
                        correct = (len(common_words) == len(expected_words)) if require_all_words else (
                                    overlap >= word_match_threshold)
                        results['correct_predictions'] += int(correct)
                        results['total_predictions'] += 1

                        # F1 and METEOR
                        f1_scores.append(calculate_f1(expected_answer, predicted_answer))
                        meteor_scores.append(calculate_meteor(expected_answer, predicted_answer))

                        # BLEU (with smoothing)
                        bleu_scores.append(
                            sentence_bleu([expected_answer.split()], predicted_answer.split())
                        )
                        print(sentence_bleu([expected_answer.split()], predicted_answer.split()))

                        # ROUGE (with empty check)
                        if predicted_answer and expected_answer:
                            rouge_scores.append(
                                rouge.get_scores(predicted_answer, expected_answer)[0]['rouge-l']['f']
                            )

                        # Semantic Similarity
                        embeddings = sentence_model.encode([expected_answer, predicted_answer])
                        similarity_scores.append(
                            cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
                        )

                        confidence_scores.append(confidence)

                    except Exception as e:
                        print(f"Skipping sample due to error: {str(e)}")
                        continue

        # Update progress
        results['processed_samples'] = i + 1

        # Checkpointing
        if (i + 1) % clear_cache_every == 0:
            # Update running averages
            results.update({
                'avg_f1': np.mean(f1_scores) if f1_scores else 0,
                'avg_meteor': np.mean(meteor_scores) if meteor_scores else 0,
                'avg_bleu': np.mean(bleu_scores) if bleu_scores else 0,
                'avg_rouge': np.mean(rouge_scores) if rouge_scores else 0,
                'avg_similarity': np.mean(similarity_scores) if similarity_scores else 0,
                'avg_confidence': np.mean(confidence_scores) if confidence_scores else 0,
                'avg_word_overlap': np.mean(word_overlap_scores) if word_overlap_scores else 0,
                'accuracy': results['correct_predictions'] / results['total_predictions'] if results[
                                                                                                 'total_predictions'] > 0 else 0,
            })

            with open(output_file, "wb") as f:
                pickle.dump(results, f)

            torch.cuda.empty_cache()
            print(f"\nCheckpoint saved at sample {i + 1}")

    # Final aggregation
    if results['total_predictions'] > 0:
        results.update({
            'avg_f1': np.mean(f1_scores) if f1_scores else 0,
            'avg_meteor': np.mean(meteor_scores) if meteor_scores else 0,
            'avg_bleu': np.mean(bleu_scores) if bleu_scores else 0,
            'avg_rouge': np.mean(rouge_scores) if rouge_scores else 0,
            'avg_similarity': np.mean(similarity_scores) if similarity_scores else 0,
            'avg_confidence': np.mean(confidence_scores) if confidence_scores else 0,
            'avg_word_overlap': np.mean(word_overlap_scores) if word_overlap_scores else 0,
            'accuracy': results['correct_predictions'] / results['total_predictions'],
        })

    # Save final results
    with open(output_file, "wb") as f:
        pickle.dump(results, f)

    print("\nEvaluation Complete!")
    for k, v in results.items():
        if isinstance(v, (int, float)):
            print(f"{k:>20}: {v:.4f}")

    return results
class CustomTrainer(SFTTrainer):
    def log(self, logs, start_time=None, **kwargs):
        super().log(logs, **kwargs)
        if "loss" in logs:
            train_losses.append(logs["loss"])


def calculate_meteor(true_answer, predicted_answer):
    # Ensure NLTK tokenization
    true_answer_tokens = word_tokenize(true_answer)  # Convert to list of words
    predicted_answer_tokens = word_tokenize(predicted_answer)  # Convert to list of words

    return meteor_score([true_answer_tokens], predicted_answer_tokens)

def calculate_f1(true_answer, predicted_answer):
    # Normalize answers: lowercase and remove punctuation
    true_answer = true_answer.lower()
    true_answer = re.sub(r'[^\w\s]', '', true_answer)
    predicted_answer = predicted_answer.lower()
    predicted_answer = re.sub(r'[^\w\s]', '', predicted_answer)

    # Tokenize into words
    true_tokens = set(true_answer.split())
    pred_tokens = set(predicted_answer.split())

    if not true_tokens or not pred_tokens:
        return 0.0

    # Calculate precision and recall
    common_tokens = true_tokens & pred_tokens
    if not common_tokens:
        return 0.0

    precision = len(common_tokens) / len(pred_tokens)
    recall = len(common_tokens) / len(true_tokens)

    # Calculate F1 score
    f1 = 2 * (precision * recall) / (precision + recall)
    return f1


# Create trainer
trainer = CustomTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,  # Add evaluation dataset
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    dataset_num_proc=2,
    args=TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        warmup_steps=20,
        max_steps=1200,
        learning_rate=1e-3,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir="outputs",
    ),
)

# Start training
trainer.train()

# Save model and tokenizer
model.save_pretrained("fine_tuned_qwen")
tokenizer.save_pretrained("fine_tuned_qwen")
with open(r"cleaned_test.json", "r", encoding="utf-8") as f:
    data2 = json.load(f)
f1=evaluate2(data2)
print(f1)



plt.figure(figsize=(10, 7))
plt.plot(steps, train_losses, label="Train Loss", color="steelblue")




plt.title('Training Loss over Steps')
plt.xlabel('Step')
plt.ylabel('Loss')
plt.legend()

# Remove internal grid lines
plt.grid(False)

plt.tight_layout()

plt.savefig("training_loss_plot.png")



























print("\nTraining completed successfully!")
print(f"Training metrics plot saved as 'training_metrics_plot.png'")


# Example usage after training:
context = "When combining orders into one control batch using QS61, it is essential to review the relevant SAP Help documentation for instructions and familiarize yourself with material specification documentation to ensure accurate information. Utilizing SAP's reporting features can also be helpful in generating detailed reports on material specifications and descriptions. Additionally, proper configuration of orders for combination in QS61 is crucial, as well as reviewing and validating the combined control batch for accuracy and completeness. To achieve this, it is recommended to implement robust validation checks to ensure data accuracy and completeness, and leveraging SAP's support resources to resolve any issues or queries related to QS61 functionality. Moreover, it is also a good practice to utilize QS61 as a tool to consolidate and manage multiple orders, while ensuring that all material specifications and descriptions are accurately captured. By following these best practices, users can ensure a successful combination of orders into one control batch using QS61."
question = "QS61 how it works?"

answer, confidence = generate_answer(context, question)
print(f"\nGenerated answer: {answer}")
print(f"Confidence score: {confidence:.4f}")

# Calculate F1 score for the example
true_answer = "access transaction V/13 to manage condition tables"
f1 = calculate_f1(true_answer, answer)
print(f"F1 score for this example: {f1:.4f}")