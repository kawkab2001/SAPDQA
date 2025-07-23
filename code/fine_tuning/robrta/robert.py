import json
import torch
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from nltk.translate.meteor_score import meteor_score
import nltk
nltk.download('wordnet')
with open('complete_finall.json', 'rb') as f:
  squad = json.load(f)
# Find the group about Greece
gr = -1
for idx, group in enumerate(squad['data']):
  # print(group['title'])
  if group['title'] == 'Greece':
    gr = idx
    #  print(gr)
    break
def read_data(path):
  # load the json file
  with open(path, 'rb') as f:
    squad = json.load(f)

  contexts = []
  questions = []
  answers = []

  for group in squad['data']:
    for passage in group['paragraphs']:
      context = passage['context']
      for qa in passage['qas']:
        question = qa['question']
        for answer in qa['answers']:
          contexts.append(context)
          questions.append(question)
          answers.append(answer)

  return contexts, questions, answers
train_contexts, train_questions, train_answers = read_data('complete_finall.json')
valid_contexts, valid_questions, valid_answers = read_data('complete_finall.json')
def add_end_idx(answers, contexts):
  for answer, context in zip(answers, contexts):
    gold_text = answer['text']
    start_idx = answer['answer_start']
    end_idx = start_idx + len(gold_text)

    # sometimes squad answers are off by a character or two so we fix this
    if context[start_idx:end_idx] == gold_text:
      answer['answer_end'] = end_idx
    elif context[start_idx-1:end_idx-1] == gold_text:
      answer['answer_start'] = start_idx - 1
      answer['answer_end'] = end_idx - 1     # When the gold label is off by one character
    elif context[start_idx-2:end_idx-2] == gold_text:
      answer['answer_start'] = start_idx - 2
      answer['answer_end'] = end_idx - 2     # When the gold label is off by two characters

add_end_idx(train_answers, train_contexts)
add_end_idx(valid_answers, valid_contexts)
from transformers import RobertaTokenizerFast, RobertaForQuestionAnswering

tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base')

train_encodings = tokenizer(train_contexts, train_questions, truncation=True, padding=True)
valid_encodings = tokenizer(valid_contexts, valid_questions, truncation=True, padding=True)
train_encodings.keys()
no_of_encodings = len(train_encodings['input_ids'])
print(f'We have {no_of_encodings} context-question pairs')
tokenizer.decode(train_encodings['input_ids'][0])
def add_token_positions(encodings, answers):
  start_positions = []
  end_positions = []
  for i in range(len(answers)):
    start_positions.append(encodings.char_to_token(i, answers[i]['answer_start']))
    end_positions.append(encodings.char_to_token(i, answers[i]['answer_end'] - 1))

    # if start position is None, the answer passage has been truncated
    if start_positions[-1] is None:
      start_positions[-1] = tokenizer.model_max_length
    if end_positions[-1] is None:
      end_positions[-1] = tokenizer.model_max_length

  encodings.update({'start_positions': start_positions, 'end_positions': end_positions})

add_token_positions(train_encodings, train_answers)
add_token_positions(valid_encodings, valid_answers)
class SQuAD_Dataset(torch.utils.data.Dataset):
  def __init__(self, encodings):
    self.encodings = encodings
  def __getitem__(self, idx):
    return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
  def __len__(self):
    return len(self.encodings.input_ids)
train_dataset = SQuAD_Dataset(train_encodings)
valid_dataset = SQuAD_Dataset(valid_encodings)
from torch.utils.data import DataLoader

# Define the dataloaders
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=16)
from transformers import BertForQuestionAnswering

model = RobertaForQuestionAnswering.from_pretrained("roberta-base")
# Check on the available device - use GPU
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(f'Working on {device}')
from transformers import AdamW


def normalize_text(s):
  """Removing articles and punctuation, and standardizing whitespace are all typical text processing steps."""
  import string, re
  def remove_articles(text):
    regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
    return re.sub(regex, " ", text)

  def white_space_fix(text):
    return " ".join(text.split())

  def remove_punc(text):
    exclude = set(string.punctuation)
    return "".join(ch for ch in text if ch not in exclude)

  def lower(text):
    return text.lower()

  return white_space_fix(remove_articles(remove_punc(lower(s))))


def exact_match(prediction, truth):
  return bool(normalize_text(prediction) == normalize_text(truth))


def compute_f1(prediction, truth):
  pred_tokens = normalize_text(prediction).split()
  truth_tokens = normalize_text(truth).split()

  # if either the prediction or the truth is no-answer then f1 = 1 if they agree, 0 otherwise
  if len(pred_tokens) == 0 or len(truth_tokens) == 0:
    return int(pred_tokens == truth_tokens)

  common_tokens = set(pred_tokens) & set(truth_tokens)

  # if there are no common tokens then f1 = 0
  if len(common_tokens) == 0:
    return 0

  prec = len(common_tokens) / len(pred_tokens)
  rec = len(common_tokens) / len(truth_tokens)

  return round(2 * (prec * rec) / (prec + rec), 2)


def get_prediction(context, question):
  # Correct order: context first, then question
  inputs = tokenizer.encode_plus(context, question, return_tensors='pt').to(device)
  outputs = model(**inputs)

  # Access start and end logits correctly
  start_logits = outputs.start_logits
  end_logits = outputs.end_logits

  answer_start = torch.argmax(start_logits)
  answer_end = torch.argmax(end_logits) + 1  # Add 1 to include the end token

  # Convert tokens to answer string
  answer = tokenizer.convert_tokens_to_string(
    tokenizer.convert_ids_to_tokens(inputs['input_ids'][0][answer_start:answer_end]))

  return answer


N_EPOCHS = 20
def compute_confidence(start_logits, end_logits):
  start_probs = torch.nn.functional.softmax(start_logits, dim=1)
  end_probs = torch.nn.functional.softmax(end_logits, dim=1)
  confidence_scores = (start_probs.max(dim=1).values * end_probs.max(dim=1).values).cpu().numpy()
  return confidence_scores
optim = AdamW(model.parameters(), lr=5e-5)

model.to(device)
model.train()

import pandas as pd
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu
from sentence_transformers import SentenceTransformer

# Load a pre-trained BERT-based embedding model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

def compute_similarity(text1, text2):
  from sklearn.metrics.pairwise import cosine_similarity

  emb1 = embedding_model.encode(text1, convert_to_tensor=True)
  emb2 = embedding_model.encode(text2, convert_to_tensor=True)

  similarity = cosine_similarity(emb1.cpu().numpy().reshape(1, -1), emb2.cpu().numpy().reshape(1, -1))
  return similarity[0][0]

avg_simlarity = []
# Initialize lists to store results for each epoch
losses = []
f1_scores = []
em_scores = []
bleu_scores = []
rouge1_scores = []

meteor_scores = []
confidence_scores = []
meteor_scores = []
confidence_scores = []
# Training loop with loss tracking
for epoch in range(N_EPOCHS):
  loop = tqdm(train_loader, leave=True)
  epoch_loss = 0
  for batch in loop:
    optim.zero_grad()
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    start_positions = batch['start_positions'].to(device)
    end_positions = batch['end_positions'].to(device)

    outputs = model(input_ids, attention_mask=attention_mask, start_positions=start_positions,end_positions=end_positions)
    loss = outputs[0]
    loss.backward()
    optim.step()

    epoch_loss += loss.item()

    loop.set_description(f'Epoch {epoch + 1}')
    loop.set_postfix(loss=loss.item())

  # Append the average loss for this epoch
  avg_loss = epoch_loss / len(train_loader)
  losses.append(avg_loss)
acc = []

# Now we evaluate the model on the validation set after training
model.eval()
for batch in tqdm(valid_loader):
  with torch.no_grad():
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    start_true = batch['start_positions'].to(device)
    end_true = batch['end_positions'].to(device)

    outputs = model(input_ids, attention_mask=attention_mask)

    start_pred = torch.argmax(outputs['start_logits'], dim=1)
    end_pred = torch.argmax(outputs['end_logits'], dim=1)
    start_acc = (start_pred == start_true).float().mean().item()
    end_acc = (end_pred == end_true).float().mean().item()
    acc.append(start_acc)
    acc.append(end_acc)

    conf_scores = compute_confidence(outputs['start_logits'], outputs['end_logits'])
    confidence_scores.extend(conf_scores)
    # Collect metrics for each batch
    for i in range(len(start_true)):
      pred_answer = get_prediction(valid_contexts[i], valid_questions[i])
      true_answer = valid_answers[i]['text']

      # Exact Match
      em_score = exact_match(pred_answer, true_answer)
      em_scores.append(em_score)

      # F1 Score
      f1_score = compute_f1(pred_answer, true_answer)
      f1_scores.append(f1_score)

      # BLEU Score
      bleu_score = sentence_bleu([true_answer.split()], pred_answer.split())
      bleu_scores.append(bleu_score)
      similarity_score = compute_similarity(pred_answer, true_answer)
      avg_simlarity.append(similarity_score)

      # ROUGE-1 Score
      scorer = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)
      rouge_score = scorer.score(true_answer, pred_answer)['rouge1'].fmeasure
      rouge1_scores.append(rouge_score)
      ref = normalize_text(true_answer).split()
      hyp = normalize_text(pred_answer).split()
      meteor = meteor_score([ref], hyp)
      meteor_scores.append(meteor)

# Calculate averages of metrics
avg_simlarity = np.mean(avg_simlarity)

avg_f1 = np.mean(f1_scores)
avg_em = np.mean(em_scores)
avg_bleu = np.mean(bleu_scores)
avg_rouge1 = np.mean(rouge1_scores)
avg_meteor = np.mean(meteor_scores)
avg_confidence = np.mean(confidence_scores)
avg_acc = sum(acc) / len(acc)
# Plot loss curve
plt.figure(figsize=(10, 6))
plt.plot(range(1, N_EPOCHS + 1), losses, marker='o', color='b', label='Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss over Epochs')
plt.grid(True)
plt.legend()
plt.savefig('robert.png')



output_dir = "./saved_model_robert"
os.makedirs(output_dir, exist_ok=True)

# Save model and tokenizer
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

acc = sum(acc)/len(acc)

print("\n\nT/P\tanswer_start\tanswer_end\n")
for i in range(len(start_true)):
  print(f"true\t{start_true[i]}\t{end_true[i]}\n"
        f"pred\t{start_pred[i]}\t{end_pred[i]}\n")



# Save metrics
metrics = {
  'F1 Score': f1_scores,
  'Exact Match': em_scores,
  'BLEU': bleu_scores,
  'ROUGE-1': rouge1_scores,
  'METEOR': meteor_scores,
  'Confidence': confidence_scores,
  'Accuracy': [avg_acc]*len(f1_scores),
  'Similarity ': avg_simlarity

}
df_metrics = pd.DataFrame(metrics)
df_metrics.to_csv('robert.csv', index=False)





print("\nFinal Metrics:")
print(f"Average F1: {avg_f1:.4f}")
print(f"Average EM: {avg_em:.4f}")
print(f"Average BLEU: {avg_bleu:.4f}")
print(f"Average ROUGE-1: {avg_rouge1:.4f}")
print(f"Average METEOR: {avg_meteor:.4f}")
print(f"Average Confidence: {avg_confidence:.4f}")
print(f"Position Accuracy: {avg_acc:.4f}")
print(f"Average Similarity: {avg_simlarity:.4f}")












































def get_prediction(context, question):
  # Correct order: context first, then question
  inputs = tokenizer.encode_plus(context, question, return_tensors='pt').to(device)
  outputs = model(**inputs)

  # Access start and end logits correctly
  start_logits = outputs.start_logits
  end_logits = outputs.end_logits

  answer_start = torch.argmax(start_logits)
  answer_end = torch.argmax(end_logits) + 1  # Add 1 to include the end token

  # Convert tokens to answer string
  answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(inputs['input_ids'][0][answer_start:answer_end]))

  return answer


def normalize_text(s):
  """Removing articles and punctuation, and standardizing whitespace are all typical text processing steps."""
  import string, re
  def remove_articles(text):
    regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
    return re.sub(regex, " ", text)
  def white_space_fix(text):
    return " ".join(text.split())
  def remove_punc(text):
    exclude = set(string.punctuation)
    return "".join(ch for ch in text if ch not in exclude)
  def lower(text):
    return text.lower()

  return white_space_fix(remove_articles(remove_punc(lower(s))))

def exact_match(prediction, truth):
    return bool(normalize_text(prediction) == normalize_text(truth))

def compute_f1(prediction, truth):
  pred_tokens = normalize_text(prediction).split()
  truth_tokens = normalize_text(truth).split()

  # if either the prediction or the truth is no-answer then f1 = 1 if they agree, 0 otherwise
  if len(pred_tokens) == 0 or len(truth_tokens) == 0:
    return int(pred_tokens == truth_tokens)

  common_tokens = set(pred_tokens) & set(truth_tokens)

  # if there are no common tokens then f1 = 0
  if len(common_tokens) == 0:
    return 0

  prec = len(common_tokens) / len(pred_tokens)
  rec = len(common_tokens) / len(truth_tokens)

  return round(2 * (prec * rec) / (prec + rec), 2)

def question_answer(context, question,answer):
  prediction = get_prediction(context,question)
  em_score = exact_match(prediction, answer)
  f1_score = compute_f1(prediction, answer)

  print(f'Question: {question}')
  print(f'Prediction: {prediction}')
  print(f'True Answer: {answer}')
  print(f'Exact match: {em_score}')
  print(f'F1 score: {f1_score}\n')

context = """When combining orders into one control batch using QS61, it is essential to review the relevant SAP Help documentation for instructions and familiarize yourself with material specification documentation to ensure accurate information. Utilizing SAP's reporting features can also be helpful in generating detailed reports on material specifications and descriptions. Additionally, proper configuration of orders for combination in QS61 is crucial, as well as reviewing and validating the combined control batch for accuracy and completeness. To achieve this, it is recommended to implement robust validation checks to ensure data accuracy and completeness, and leveraging SAP's support resources to resolve any issues or queries related to QS61 functionality. Moreover, it is also a good practice to utilize QS61 as a tool to consolidate and manage multiple orders, while ensuring that all material specifications and descriptions are accurately captured. By following these best practices, users can ensure a successful combination of orders into one control batch using QS61."""


questions = ["QS61 how it works"]

answers = ["QS61 is a tool to consolidate and manage multiple orders, combining into one control batch for material specifications and descriptions. It can combine multiple orders into one control batch."]

for question, answer in zip(questions, answers):
  question_answer(context, question, answer)



