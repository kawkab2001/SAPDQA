from transformers import pipeline, T5Tokenizer, T5ForConditionalGeneration
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
from sklearn.metrics import f1_score, accuracy_score
import torch
from nltk.translate.meteor_score import meteor_score
import numpy as np
# Load pre-trained QA models for BERT and RoBERTa
qa_pipeline = pipeline("question-answering", model="deepset/bert-base-cased-squad2")
qa_pipeline2 = pipeline("question-answering", model="deepset/roberta-base-squad2")

# Load pre-trained T5 tokenizer and model for direct usage
t5_tokenizer = T5Tokenizer.from_pretrained("google/t5-small-ssm-nq")
t5_model = T5ForConditionalGeneration.from_pretrained("google/t5-small-ssm-nq")

import torch.nn.functional as F


def calculate_t5_confidence(t5_answer):
    # Tokenize the input using T5 tokenizer
    inputs = t5_tokenizer(t5_answer, return_tensors="pt", add_special_tokens=True)

    # Generate decoder input ids (usually just the start token for generation)
    decoder_input_ids = torch.tensor([[t5_tokenizer.pad_token_id]])  # Start with the pad token ID

    # Get the model's logits by passing the input through the T5 model
    with torch.no_grad():
        outputs = t5_model(input_ids=inputs["input_ids"], decoder_input_ids=decoder_input_ids)
        logits = outputs.logits  # Shape: (batch_size, sequence_length, vocab_size)

    # Apply softmax to convert logits to probabilities
    probs = F.softmax(logits, dim=-1)  # Shape: (batch_size, sequence_length, vocab_size)

    # Get the token IDs of the generated answer
    generated_token_ids = t5_tokenizer.encode(t5_answer, return_tensors="pt", add_special_tokens=True)

    # Ensure the generated token IDs are within the bounds of the probs tensor
    if generated_token_ids.size(1) > probs.size(1):
        # If the generated sequence is longer than the model's output, truncate it
        generated_token_ids = generated_token_ids[:, :probs.size(1)]

    # Calculate the confidence as the mean probability of the generated tokens
    try:
        confidence = probs[0, torch.arange(generated_token_ids.size(1)), generated_token_ids[0]].mean().item()
    except IndexError:
        # Handle cases where the generated sequence is empty or misaligned
        confidence = 0.0  # Default confidence for invalid cases

    return confidence

# Function to evaluate answers
from nltk.tokenize import word_tokenize
import nltk
nltk.download("punkt")
def evaluate_answer(predicted, reference):
    # Tokenize the predicted and reference answers
    predicted_tokens = word_tokenize(predicted)
    reference_tokens = word_tokenize(reference)

    # Compute BLEU Score
    bleu_score = sentence_bleu([reference_tokens], predicted_tokens)

    # Compute ROUGE Score
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    rouge_scores = scorer.score(reference, predicted)

    # Compute METEOR Score
    meteor = meteor_score([reference_tokens], predicted_tokens)

    return bleu_score, rouge_scores, meteor

def calculate_bert_confidence(answer, model_output):
    # Extract the start and end positions from the model's output
    start_index = model_output["start"]
    end_index = model_output["end"]

    # Confidence is based on the length of the answer (normalized)
    # You can also use other heuristics, such as the probability of the answer span
    confidence = 1.0 / (end_index - start_index + 1)  # Example heuristic

    return confidence

import re

def normalize_answer(answer):
    """Normalize the answer by lowercasing, removing extra spaces, and stripping punctuation."""
    answer = answer.lower().strip()
    answer = re.sub(r'\s+', ' ', answer)  # Replace multiple spaces with a single space
    answer = re.sub(r'[^\w\s]', '', answer)  # Remove punctuation
    return answer
def calculate_roberta_confidence(answer, model_output):
    # Extract the start and end positions from the model's output
    start_index = model_output["start"]
    end_index = model_output["end"]

    # Confidence is based on the length of the answer (normalized)
    # You can also use other heuristics, such as the probability of the answer span
    confidence = 1.0 / (end_index - start_index + 1)  # Example heuristic

    return confidence
f1_bertt=[]
f1_robert=[]
f1_t55=[]

blue_bert=[]
blue_robert=[]
blue_t55=[]

rouge_bert=[]
rouge_robert=[]
rouge_t5=[]

meteor_bert=[]
meteor_robert=[]
meteor_t5=[]

accuracy_bertt=[]
accuracy_robertt=[]
accuracy_t55=[]

confidence_bert=[]
confidence_robert=[]
confidence_t5=[]


sim_bert=[]
sim_robert=[]
sim_t5=[]
def calculate_accuracy(a, b):
    # Split the strings into words
    a_words = set(a.split())
    b_words = set(b.split())

    # Calculate the intersection (common words)
    common_words = a_words.intersection(b_words)

    # Accuracy = (Number of common words) / (Total number of unique words)
    total_words = len(a_words.union(b_words))

    if total_words == 0:  # To avoid division by zero
        return 0.0

    accuracy = len(common_words) / total_words
    return accuracy
from sklearn.metrics import f1_score, accuracy_score
import numpy as np
from sentence_transformers import SentenceTransformer

# Load a pre-trained BERT-based embedding model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

def compute_similarity(text1, text2):
  from sklearn.metrics.pairwise import cosine_similarity

  emb1 = embedding_model.encode(text1, convert_to_tensor=True)
  emb2 = embedding_model.encode(text2, convert_to_tensor=True)

  similarity = cosine_similarity(emb1.cpu().numpy().reshape(1, -1), emb2.cpu().numpy().reshape(1, -1))
  return similarity[0][0]
def tes(question, context, real_answer):
    # Get the answers from each model
    bert_output = qa_pipeline(question=question, context=context)
    roberta_output = qa_pipeline2(question=question, context=context)

    # Extract predicted answers
    bert_answer = bert_output["answer"]
    roberta_answer = roberta_output["answer"]

    # For T5, use text generation to get the answer
    input_text = f"question: {question} context: {context}"
    t5_input = t5_tokenizer.encode(input_text, return_tensors="pt", add_special_tokens=True)
    t5_answer_ids = t5_model.generate(t5_input)  # Generate answer ids from the model
    t5_answer_text = t5_tokenizer.decode(t5_answer_ids[0], skip_special_tokens=True)

    # Normalize answers
    normalized_bert_answer = normalize_answer(bert_answer)
    normalized_roberta_answer = normalize_answer(roberta_answer)
    normalized_t5_answer = normalize_answer(t5_answer_text)
    normalized_real_answer = normalize_answer(real_answer)

    # Calculate confidence scores
    bert_confidence = calculate_bert_confidence(bert_answer, bert_output)
    roberta_confidence = calculate_roberta_confidence(roberta_answer, roberta_output)
    t5_confidence = calculate_t5_confidence(t5_answer_text)

    # Evaluate each model
    bert_bleu, bert_rouge, bert_meteor = evaluate_answer(bert_answer, real_answer)
    roberta_bleu, roberta_rouge, roberta_meteor = evaluate_answer(roberta_answer, real_answer)
    t5_bleu, t5_rouge, t5_meteor = evaluate_answer(t5_answer_text, real_answer)

    # Compute F1 and Accuracy
    # For F1, tokenize the answers and compare at the token level
    bert_tokens = word_tokenize(normalized_bert_answer)
    roberta_tokens = word_tokenize(normalized_roberta_answer)
    t5_tokens = word_tokenize(normalized_t5_answer)
    real_tokens = word_tokenize(normalized_real_answer)

    # Create binary vectors for F1 calculation
    def create_binary_vector(tokens, real_tokens):
        return [1 if token in real_tokens else 0 for token in tokens]

    bert_binary = create_binary_vector(bert_tokens, real_tokens)
    roberta_binary = create_binary_vector(roberta_tokens, real_tokens)
    t5_binary = create_binary_vector(t5_tokens, real_tokens)
    real_binary = [1] * len(real_tokens)  # All tokens in the real answer are considered correct

    # Pad or truncate binary vectors to ensure equal length
    max_length = max(len(real_binary), len(bert_binary), len(roberta_binary), len(t5_binary))
    def pad_or_truncate(binary, max_length):
        if len(binary) < max_length:
            return binary + [0] * (max_length - len(binary))  # Pad with zeros
        else:
            return binary[:max_length]  # Truncate to max_length

    real_binary = pad_or_truncate(real_binary, max_length)
    bert_binary = pad_or_truncate(bert_binary, max_length)
    roberta_binary = pad_or_truncate(roberta_binary, max_length)
    t5_binary = pad_or_truncate(t5_binary, max_length)

    # Compute F1 scores
    f1_bert = f1_score(real_binary, bert_binary, average="macro", zero_division=1)
    f1_roberta = f1_score(real_binary, roberta_binary, average="macro", zero_division=1)
    f1_t5 = f1_score(real_binary, t5_binary, average="macro", zero_division=1)

    # Compute Accuracy (binary: 1 if the entire answer matches, 0 otherwise)
    accuracy_bert = calculate_accuracy(bert_answer, real_answer)
    accuracy_roberta = calculate_accuracy(roberta_answer, real_answer)
    accuracy_t5 = calculate_accuracy(t5_answer_text, real_answer)


    sb=compute_similarity(bert_answer, real_answer)
    sr = compute_similarity(roberta_answer, real_answer)
    st = compute_similarity(t5_answer_text, real_answer)
    sim_bert.append(sb)
    sim_robert.append(sr)
    sim_t5.append(st)
    # Store results for later analysis
    f1_bertt.append(f1_bert)
    f1_robert.append(f1_roberta)
    f1_t55.append(f1_t5)

    blue_bert.append(bert_bleu)
    blue_robert.append(roberta_bleu)
    blue_t55.append(t5_bleu)

    rouge_bert.append(bert_rouge)
    rouge_robert.append(roberta_rouge)
    rouge_t5.append(t5_rouge)

    meteor_bert.append(bert_meteor)
    meteor_robert.append(roberta_meteor)
    meteor_t5.append(t5_meteor)

    accuracy_bertt.append(accuracy_bert)
    accuracy_robertt.append(accuracy_roberta)
    accuracy_t55.append(accuracy_t5)

    confidence_bert.append(bert_confidence)
    confidence_robert.append(roberta_confidence)
    confidence_t5.append(t5_confidence)

    # Print the results
    print("\nBERT Answer:", bert_answer)
    print("BERT Confidence:", bert_confidence)
    print("BERT BLEU Score:", bert_bleu)
    print("BERT ROUGE Scores:", bert_rouge["rouge1"].fmeasure)
    print("BERT METEOR Score:", bert_meteor)
    print("BERT F1 Score:", f1_bert)
    print("BERT Accuracy:", accuracy_bert)

    print("\nRoBERTa Answer:", roberta_answer)
    print("RoBERTa Confidence:", roberta_confidence)
    print("RoBERTa BLEU Score:", roberta_bleu)
    print("RoBERTa ROUGE Scores:", roberta_rouge["rouge1"].fmeasure)
    print("RoBERTa METEOR Score:", roberta_meteor)
    print("RoBERTa F1 Score:", f1_roberta)
    print("RoBERTa Accuracy:", accuracy_roberta)

    print("\nT5 Answer:", t5_answer_text)
    print("T5 Confidence:", t5_confidence)
    print("T5 BLEU Score:", t5_bleu)
    print("T5 ROUGE Scores:", t5_rouge["rouge1"].fmeasure)
    print("T5 METEOR Score:", t5_meteor)
    print("T5 F1 Score:", f1_t5)
    print("T5 Accuracy:", accuracy_t5)
import json
with open('complete.json', 'r', encoding='latin1') as f:
    data = json.load(f)

tes("How to delete records from SSCUI", "To delete a customized test record created in SSCUI (100297) before transport, it is essential to follow a specific procedure to ensure the deletion process goes smoothly and securely. First, activating the object by setting its status to \"Activated\" is a critical step that enables the necessary technical procedures for deletion. However, due to the sensitive nature of data management, this task should be approached with caution, as deleting records requires specialized knowledge to avoid any potential data loss or corruption. Given the complexity and importance of the task, it's crucial to seek professional assistance from SAP support via an incident report to ensure that the record is deleted in a secure and controlled manner. This ensures that all necessary precautions are taken to prevent any unintended consequences on the system or its data. With careful planning, technical expertise, and the right support, deleting customized test records created in SSCUI can be achieved successfully before transport", "Seek professional assistance from SAP support via an incident report to ensure a secure and controlled deletion")

for item in data['data']:
    for paragraph in item['paragraphs']:
        for qa in paragraph['qas']:
            for answer in qa['answers']:
                question =  item['title']
                context = paragraph['context']
                real_answer = answer['text']
                tes(question, context, real_answer)

f1_bmn=np.mean(f1_bertt)
f1_rmn=np.mean(f1_robert)
f1_tmn=np.mean(f1_t55)
blue_bmn=np.mean(blue_bert)
blue_rmn=np.mean(blue_robert)
blue_tmn=np.mean(blue_t55)


rouge_bert_f1 = [score["rouge1"].fmeasure for score in rouge_bert]
rouge_robert_f1 = [score["rouge1"].fmeasure for score in rouge_robert]
rouge_t5_f1 = [score["rouge1"].fmeasure for score in rouge_t5]

# Compute mean ROUGE F1 scores
rouge_bmn = np.mean(rouge_bert_f1)
rouge_rmn = np.mean(rouge_robert_f1)
rouge_tmn = np.mean(rouge_t5_f1)


acc_bmn=np.mean(accuracy_bertt)
acc_rmn=np.mean(accuracy_robertt)
acc_tmn=np.mean(accuracy_t55)


meteor_bmn=np.mean(meteor_bert)
meteor_rmn=np.mean(meteor_robert)
meteor_tmn=np.mean(meteor_t5)

con_bmn=np.mean(confidence_bert)
con_rmn=np.mean(confidence_robert)
con_tmn=np.mean(confidence_t5)

sim_bmn=np.mean(sim_bert)
sim_rmn=np.mean(sim_robert)
sim_tmn=np.mean(sim_t5)
print(f"sim_bmn: {sim_bmn}")
print(f"sim_rmn: {sim_rmn}")
print(f"sim_tmn: {sim_tmn}")

print(f"f1_bmn: {f1_bmn}")
print(f"f1_rmn: {f1_rmn}")
print(f"f1_tmn: {f1_tmn}")
print(f"blue_bmn: {blue_bmn}")
print(f"blue_rmn: {blue_rmn}")
print(f"blue_tmn: {blue_tmn}")
print(f"rouge_bmn: {rouge_bmn}")
print(f"rouge_rmn: {rouge_rmn}")
print(f"rouge_tmn: {rouge_tmn}")
print(f"acc_bmn: {acc_bmn}")
print(f"acc_rmn: {acc_rmn}")
print(f"acc_tmn: {acc_tmn}")
print(f"meteor_bmn: {meteor_bmn}")
print(f"meteor_rmn: {meteor_rmn}")
print(f"meteor_tmn: {meteor_tmn}")
print(f"con_bmn: {con_bmn}")
print(f"con_rmn: {con_rmn}")
print(f"con_tmn: {con_tmn}")
















#####################


#for qwen




########################










import json
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch
from nltk.translate.bleu_score import sentence_bleu
from rouge import Rouge
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import nltk
from nltk.translate.meteor_score import meteor_score

# Download required NLTK data
nltk.download('wordnet')
nltk.download('omw-1.4')

# Initialize sentence transformer for similarity
bert_model = SentenceTransformer('all-MiniLM-L6-v2')
import json
with open('complete_copy.json', 'r', encoding='latin1') as f:
    data = json.load(f)
# Step 2: Load the model and tokenizer
model_name = "unsloth/Qwen2-0.5B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForQuestionAnswering.from_pretrained(model_name)

# Initialize metrics lists
f1_scores = []
bleu_scores = []
rouge_scores = []
meteor_scores = []
accuracy_scores = []
confidence_scores = []
similarity_scores = []

# Initialize ROUGE scorer
rouge = Rouge()


def calculate_f1(pred_tokens, true_tokens):
    common_tokens = set(pred_tokens) & set(true_tokens)
    if len(common_tokens) == 0:
        return 0.0

    precision = len(common_tokens) / len(pred_tokens)
    recall = len(common_tokens) / len(true_tokens)
    f1 = 2 * (precision * recall) / (precision + recall)
    return f1


def calculate_accuracy(pred_text, true_text):
    return 1.0 if pred_text.strip().lower() == true_text.strip().lower() else 0.0


def calculate_confidence(start_logits, end_logits):
    start_prob = torch.softmax(start_logits, dim=-1)
    end_prob = torch.softmax(end_logits, dim=-1)
    start_conf = torch.max(start_prob).item()
    end_conf = torch.max(end_prob).item()
    return (start_conf + end_conf) / 2


def calculate_similarity(text1, text2):
    embeddings = bert_model.encode([text1, text2])
    similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
    return similarity


# Step 3: Define a function to answer questions
def answer_question(question, context, true_answer=None):
    # Tokenize input
    inputs = tokenizer.encode_plus(question, context, return_tensors="pt", max_length=512, truncation=True)

    # Get model predictions
    with torch.no_grad():
        outputs = model(**inputs)

    # Calculate confidence
    confidence = calculate_confidence(outputs.start_logits, outputs.end_logits)

    # Extract answer start and end positions
    answer_start = torch.argmax(outputs.start_logits)
    answer_end = torch.argmax(outputs.end_logits) + 1

    # Decode the answer
    answer = tokenizer.convert_tokens_to_string(
        tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][answer_start:answer_end])
    )

    # Calculate metrics if true answer is provided
    if true_answer:
        # Tokenize texts for metrics
        pred_tokens = answer.split()
        true_tokens = true_answer.split()

        # F1 Score
        f1 = calculate_f1(pred_tokens, true_tokens)
        f1_scores.append(f1)

        # BLEU Score
        bleu = sentence_bleu([true_tokens], pred_tokens)
        bleu_scores.append(bleu)

        # ROUGE Score
        try:
            rouge_score = rouge.get_scores(answer, true_answer)[0]['rouge-l']['f']
            rouge_scores.append(rouge_score)
        except:
            rouge_scores.append(0.0)

        # METEOR Score
        meteor = meteor_score([true_tokens], pred_tokens)
        meteor_scores.append(meteor)

        # Accuracy
        accuracy = calculate_accuracy(answer, true_answer)
        accuracy_scores.append(accuracy)

        # Confidence
        confidence_scores.append(confidence)

        # Similarity
        similarity = calculate_similarity(answer, true_answer)
        similarity_scores.append(similarity)

    return answer, confidence


# Step 4: Iterate over the dataset and answer questions
for article in data["data"]:
    for paragraph in article["paragraphs"]:
        context = paragraph["context"]
        for qa in paragraph["qas"]:
            question = qa["question"]
            true_answer = qa["answers"][0]["text"]

            print(f"Question: {question}")
            answer, confidence = answer_question(question, context, true_answer)
            print(f"Answer: {answer}")
            print(f"True Answer: {true_answer}")
            print(f"Confidence: {confidence:.4f}\n")

# Calculate average metrics
if f1_scores:
    print("\nEvaluation Metrics:")
    print(f"Average F1 Score: {np.mean(f1_scores):.4f}")
    print(f"Average BLEU Score: {np.mean(bleu_scores):.4f}")
    print(f"Average ROUGE-L F1: {np.mean(rouge_scores):.4f}")
    print(f"Average METEOR Score: {np.mean(meteor_scores):.4f}")
    print(f"Average Accuracy: {np.mean(accuracy_scores):.4f}")
    print(f"Average Confidence: {np.mean(confidence_scores):.4f}")
    print(f"Average Similarity: {np.mean(similarity_scores):.4f}")