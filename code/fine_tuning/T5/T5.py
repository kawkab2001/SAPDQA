
import json
import random
import string
import time
from functools import partial

import evaluate
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from datasets import Dataset
from peft import get_peft_model, LoraConfig, TaskType, PeftModel
from sentence_transformers import CrossEncoder
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoModel,
    AutoTokenizer,
    AutoModelForQuestionAnswering,
    get_scheduler,
)

# Set the seed for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed()
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

import json
from datasets import Dataset

# Load your custom dataset from the JSON file
with open('complete_finall.json', 'rb') as f:
    data = json.load(f)

transformed_data = []

for entry in data['data']:
  title = entry['title']

  for paragraph in entry['paragraphs']:
    context = paragraph['context']

    for qa in paragraph['qas']:
      question = qa['question']
      answers = [answer['text'] for answer in qa['answers']]
      answer_start = [answer['answer_start'] for answer in qa['answers']]
      answer_end = [answer['answer_end'] for answer in qa['answers']]

      # Append to transformed data
      transformed_data.append({
        'id': qa['id'],
        'title': title,
        'context': context,
        'question': question,
        'answers': {
          'text': answers,
          'answer_start': answer_start,
          'answer_end': answer_end
        }
      })

from datasets import Dataset
import pandas as pd

# Convert the DataFrame to a Dataset
transformed_data = Dataset.from_pandas(pd.DataFrame(transformed_data))

# Shuffle the dataset first
shuffled_dataset = transformed_data.shuffle(seed=42)

# Then split the dataset
squad = shuffled_dataset.train_test_split(test_size=0.2)



tokenizer = AutoTokenizer.from_pretrained("t5-small", use_fast=True)


def tokenize_preprocess(examples):
  questions = [q.strip() for q in examples["question"]]

  # Tokenize with fixed padding to max_length
  inputs = tokenizer(
    questions,
    examples["context"],
    truncation=True,
    padding="max_length",
    max_length=384,
    return_tensors="pt"
  )

  answers = examples["answers"]
  start_token_list = []
  end_token_list = []

  for i in range(len(answers)):
    answer = answers[i]
    start_idx = answer["answer_start"][0]
    end_idx = answer["answer_end"][0]

    # Convert character indices to token indices
    start_token = inputs.char_to_token(i, start_idx, sequence_index=1)
    end_token = inputs.char_to_token(i, end_idx - 1, sequence_index=1)

    # Handle truncation (positions beyond max_length are set to max_length - 1)
    if start_token is None:
      start_token = 383  # 384 - 1 (0-based index)
    if end_token is None:
      end_token = 383

    start_token_list.append(start_token)
    end_token_list.append(end_token)

  # Add sequence_ids to inputs
  batch_size = inputs["input_ids"].shape[0]
  seq_ids = []
  for i in range(batch_size):
    seq = inputs.sequence_ids(i)  # Get sequence IDs for each example
    # Replace None (padding/special tokens) with -1
    converted_seq = [s if s is not None else -1 for s in seq]
    seq_ids.append(converted_seq)

  inputs["sequence_ids"] = seq_ids  # Add sequence_ids as a list

  # Add start and end positions
  inputs["start_positions"] = start_token_list
  inputs["end_positions"] = end_token_list

  return inputs


# Map preprocessing function to the dataset (approx 2.5 minutes)
# Map preprocessing function to the dataset
tokenized_squad = squad.map(
    tokenize_preprocess,
    batched=True,
    remove_columns=squad["train"].column_names  # Remove only original columns
)

# Verify that sequence_ids is present
print(tokenized_squad["train"][0].keys())

# Format dataset to use tensors rather than native Python lists
tokenized_squad = tokenized_squad.with_format("torch")
# Define dataloaders
BATCH_SIZE = 8


from torch.nn.utils.rnn import pad_sequence

def collate_fn(batch):
    input_ids = [item['input_ids'] for item in batch]
    attention_mask = [item['attention_mask'] for item in batch]
    start_positions = [item['start_positions'] for item in batch]
    end_positions = [item['end_positions'] for item in batch]

    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)
    start_positions = torch.tensor(start_positions)
    end_positions = torch.tensor(end_positions)

    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'start_positions': start_positions,
        'end_positions': end_positions
    }

train_dataloader = DataLoader(
    tokenized_squad["train"], shuffle=True, batch_size=BATCH_SIZE, collate_fn=collate_fn
)

val_dataloader = DataLoader(
    tokenized_squad["test"], shuffle=True, batch_size=BATCH_SIZE, collate_fn=collate_fn
)


EPOCHS=2
model = AutoModelForQuestionAnswering.from_pretrained("t5-small")
model.to(device)



# Optimizer and scheduler
# Initialize optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

num_training_steps = EPOCHS * len(train_dataloader)
scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=len(train_dataloader) * 0.5,  # warmup = half an epoch
    num_training_steps=num_training_steps,
)
# Training loop (same as your existing code)

train_losses = []
val_losses = []
def train(model, train_dataloader, val_dataloader, device, optimizer, scheduler):
  """
  Trains the specified model.
  """


  model.train()

  for epoch in range(EPOCHS):
    # print(f"Epoch {epoch}, Learning rate: {scheduler.get_last_lr()[0]:.5f}")
    epoch_loss = []
    total_loss = 0
    for step, batch in enumerate(train_dataloader):
      start = time.time()
      optimizer.zero_grad()

      input_ids = batch["input_ids"].to(device)
      attention_mask = batch["attention_mask"].to(device)
      start_positions = batch["start_positions"].to(device)
      end_positions = batch["end_positions"].to(device)

      outputs = model(
        input_ids,
        attention_mask=attention_mask,
        start_positions=start_positions,
        end_positions=end_positions,
      )

      # outputs has keys 'loss', 'start_logits', 'end_logits', 'encoder_last_hidden_state'
      loss = outputs.loss
      total_loss += loss.item()

      loss.backward()
      optimizer.step()
      scheduler.step()
      stop = time.time()

      if step % 400 == 0:  # note this will run on the first step also
        train_losses.append(loss.item())
        # Calculate validation loss on a single batch only for compute reasons
        val_batch = next(iter(val_dataloader))
        with torch.no_grad():
          model.eval()
          output = model(
            val_batch["input_ids"].to(device),
            attention_mask=val_batch["attention_mask"].to(device),
            start_positions=val_batch["start_positions"].to(device),
            end_positions=val_batch["end_positions"].to(device),
          )
          val_loss = output.loss
          val_losses.append(val_loss.item())
        model.train()

        # Print statistics
        print(
          f"Epoch: {epoch + 1}/{EPOCHS} | Step: {step}/{len(train_dataloader)} | Train Loss: {loss.item():.5f} | Val Loss: {val_loss.item():.5f} |",
          f"LR: {scheduler.get_last_lr()[0]:.5f} | Time of Last Batch: {stop - start:.2f} \n",
        )

    avg_train_loss = total_loss / len(train_dataloader)
    epoch_loss.append(avg_train_loss)

    print(f"Epoch {epoch}, Train Loss: {avg_train_loss}")


train(model, train_dataloader, val_dataloader, device, optimizer, scheduler)
model.save_pretrained("./saved_model_lora_T5_simple")
tokenizer.save_pretrained("./saved_model_lora_T5_simple")
plt.figure(figsize=(12, 8))
plt.title("Training Loss")
plt.plot(train_losses, label="Train")
plt.xlabel("Step")
plt.ylabel("Loss")
plt.xticks(
    ticks=plt.xticks()[0][1:], labels=400 * np.array(plt.xticks()[0][1:], dtype=int)
)  # steps * 400
plt.legend()
plt.show()
plt.savefig('t5_simple.png')




n_best = 10  # number of start and end indices to consider as candidates (no need to check all 384 logits)
max_answer_len = 30


def retrieve_preds_and_labels(
        start_logits,
        end_logits,
        input_ids,
        seq_ids,
        start_pos=None,
        end_pos=None,
        n_best=10,
        max_answer_len=30,
        inference=False,
):
  """
  Mapping helper function which post-processes and decodes fine-tuned T5ForQuestionAnswering model outputs
  as well as decoded ground truth labels.

  Inputs:
    - start_logits and end_logits: Model output logits for start and end positions (tensors of length `seq_len`).
    - input_ids: Tokenized input IDs (tensor of length `seq_len`).
    - seq_ids: Sequence IDs indicating which tokens belong to the context (tensor of length `seq_len`).
    - start_pos and end_pos: Token indices of the ground truth labels (tensors of length 1).
    - n_best (int): Number of start and end indices to consider as candidates.
    - max_answer_len (int): Maximum token length of a predicted answer.
    - inference (bool): If True, only processes predictions (no labels).

  Returns:
    - If inference=False: Tuple of (decoded predictions, decoded ground truth labels).
    - If inference=True: Tuple containing only decoded predictions.
  """
  assert (
          isinstance(n_best, int)
          and isinstance(max_answer_len, int)
          and n_best > 0
          and max_answer_len > 0
  )

  # Get the top `n_best` start and end indices
  start_idx_list = np.argsort(start_logits.cpu().numpy())[-1: (-n_best - 1): -1]
  end_idx_list = np.argsort(end_logits.cpu().numpy())[-1: (-n_best - 1): -1]

  valid_answers = []
  for start_idx in start_idx_list:
    for end_idx in end_idx_list:
      # Ignore out-of-scope answers (indices outside the context)
      if (
              seq_ids[start_idx].item() != 1
              or seq_ids[end_idx].item() != 1
      ):
        continue
      # Ignore answers with negative length or > max_answer_len
      if start_idx > end_idx or end_idx - start_idx + 1 > max_answer_len:
        continue

      # If this start-end pair is valid, add it to the list
      valid_answers.append(
        {
          "score": start_logits[start_idx] + end_logits[end_idx],
          "start_idx": start_idx,
          "end_idx": end_idx,
        }
      )

  # Handle the case where no valid answers are found
  if not valid_answers:
    final_decoded_preds = ""  # Return an empty string as the prediction
  else:
    # Take the prediction with the highest score
    final_preds = sorted(valid_answers, key=lambda x: x["score"], reverse=True)[0]
    final_decoded_preds = tokenizer.decode(
      input_ids[final_preds["start_idx"]: (final_preds["end_idx"] + 1)]
    )

  # Decode ground truth labels if not in inference mode
  if not inference:
    labels = tokenizer.decode(input_ids[start_pos: (end_pos + 1)])
    return final_decoded_preds, labels
  else:
    return (final_decoded_preds,)




def postprocess(batch, output, inference=False, **kwargs):
  """
  Postprocesses and decodes model output (and ground truth labels if any).
  """
  # Debug: Check keys in the batch
  print("Keys in batch:", batch.keys())

  # batch size used
  b_size = batch["input_ids"].size(0)

  # prepare map function with fixed inference and keyword arguments
  mapfunc = partial(retrieve_preds_and_labels, inference=inference, **kwargs)

  # if inference, no start/end positions, and we initialize placeholder tensors
  if inference:
    start_pos, end_pos = torch.empty((b_size, 1)), torch.empty((b_size, 1))
  else:
    start_pos, end_pos = batch["start_positions"], batch["end_positions"]

  # map helper function
  postprocessed_output = list(
    map(
      mapfunc,
      output.start_logits,
      output.end_logits,
      batch["input_ids"],
      batch["sequence_ids"],  # Ensure sequence_ids is accessed here
      start_pos,
      end_pos,
    )
  )

  # output shape above: list of length `batch_size` of 2-tuples (pred, label) or 1-tuple (pred, )
  preds = np.array([postprocessed_output[i][0] for i in range(b_size)])
  if not inference:
    labels = np.array([postprocessed_output[i][1] for i in range(b_size)])
    return preds, labels
  else:
    return preds


def normalization(text):
  """
  Normalizes a given text by fixing whitespaces, converting to lowercase, and removing punctuation.
  This function does not remove stopwords, articles, or translate numbers to words as these actions
  can affect the length of the strings and thus the F-1 score.

  Input:
      -text (str): Text string to be normalized
  Returns:
      -The normalized text string
  """
  # Fix whitespaces, convert lowercase
  text = " ".join(text.split()).lower()

  # Remove punctuation
  text = text.translate(str.maketrans("", "", string.punctuation))

  return text





def exact_match(preds, labels):

  # Normalize predictions and labels
  preds = np.vectorize(normalization)(preds)
  labels = np.vectorize(normalization)(labels)

  return np.mean(preds == labels)





def f1(preds, labels):
  """
  Computes F-1 score word-level.

  Input:
      -preds (np.array): Array of prediction strings
      -labels (np.array): Array of label strings
  Returns:
      -Mean F-1 score (float) for all pairs of normalized predictions and labels
  """

  f1_list = []

  # Normalize predictions and labels
  preds = np.vectorize(normalization)(preds)
  labels = np.vectorize(normalization)(labels)

  # Calculates F-1 Score for each pair of preds & labels
  for i in range(len(preds)):
    pred_tokens = preds[i].split()
    act_tokens = labels[i].split()

    common_tokens = set(pred_tokens) & set(act_tokens)
    if len(common_tokens) == 0:
      f1_list.append(0)
    else:
      pre = len(common_tokens) / len(pred_tokens)
      rec = len(common_tokens) / len(act_tokens)
      f1 = 2 * (pre * rec) / (pre + rec)
      f1_list.append(f1)

  return np.mean(f1_list)
def sas(cross_encoder, preds, labels):
  """
     Computes Semantic Answer Similarity (SAS) scores between predictions and labels via a cross-encoder.

     Input:
         -cross_encoder: Cross-encoder model used for prediction
         -preds (np.array): Array of prediction strings
         -labels (np.array): Array of label strings
     Returns:
         -Mean SAS score (float) for all prediction-label pairs

     """
  cross_encoder_input = [(preds[i], labels[i]) for i in range(len(preds))]
  sas_scores = cross_encoder.predict(cross_encoder_input)

  return sas_scores.mean()



# Load the model and tokenizer
model = AutoModelForQuestionAnswering.from_pretrained("./saved_model_lora_T5_simple")
tokenizer = AutoTokenizer.from_pretrained("./saved_model_lora_T5_simple")

# Move the model to the GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define the context and question for testing
context = "When combining orders into one control batch using QS61, it is essential to review the relevant SAP Help documentation for instructions on how to process orders efficiently and ensure accuracy."
question = "QS61 how it works"

# Tokenize the input question and context
inputs = tokenizer(question, context, add_special_tokens=True, return_tensors="pt").to(device)

# Perform inference (model outputs)
with torch.no_grad():
    outputs = model(**inputs)

# Get the start and end position of the answer
start_index = outputs.start_logits.argmax()
end_index = outputs.end_logits.argmax()

# Get the answer from the context
answer_tokens = inputs.input_ids[0][start_index:end_index + 1]
answer = tokenizer.decode(answer_tokens, skip_special_tokens=True)

print(f"Answer: {answer}")