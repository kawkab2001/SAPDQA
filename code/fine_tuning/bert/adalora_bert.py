import json
import random
import string
import time
from functools import partial
import pandas as pd
import evaluate
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from datasets import Dataset
from peft import get_peft_model, AdaLoraConfig, TaskType, PeftModel
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

# Load your custom dataset from the JSON file
with open('complete.json', 'rb') as f:
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

# Convert the DataFrame to a Dataset
transformed_data = Dataset.from_pandas(pd.DataFrame(transformed_data))

# Shuffle the dataset first
shuffled_dataset = transformed_data.shuffle(seed=42)

# Then split the dataset
squad = shuffled_dataset.train_test_split(test_size=0.2)

# Use BERT tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", use_fast=True)

def tokenize_preprocess(examples):
    questions = [q.strip() for q in examples["question"]]

    # Tokenize with truncation and padding
    inputs = tokenizer(
        questions,
        examples["context"],
        truncation=True,
        padding=True,
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

        # Get start and end token indices from the character indices
        start_token = inputs.char_to_token(i, start_idx, sequence_index=1)
        end_token = inputs.char_to_token(i, end_idx - 1, sequence_index=1)

        # Handle None cases where char_to_token returns None (due to truncation)
        if start_token is None:
            start_token = min(len(inputs["input_ids"][0]), 384)  # Set to max token length if truncated
        if end_token is None:
            end_token = min(len(inputs["input_ids"][0]), 384)  # Set to max token length if truncated

        start_token_list.append(start_token)
        end_token_list.append(end_token)

    # Append sequence_ids necessary for post-processing
    map_batch_size = len(inputs["input_ids"])
    seq_ids = [inputs.sequence_ids(i) for i in range(map_batch_size)]

    inputs["start_positions"] = start_token_list
    inputs["end_positions"] = end_token_list
    inputs["sequence_ids"] = seq_ids

    return inputs

# Map preprocessing function to the dataset
tokenized_squad = squad.map(
    tokenize_preprocess, batched=True, remove_columns=squad["train"].column_names
)

# Format dataset to use tensors rather than native Python lists
tokenized_squad = tokenized_squad.with_format("torch")

# Define dataloaders
BATCH_SIZE = 10

train_dataloader = DataLoader(
    tokenized_squad["train"], shuffle=True, batch_size=BATCH_SIZE
)

val_dataloader = DataLoader(
    tokenized_squad["test"], shuffle=True, batch_size=BATCH_SIZE
)

EPOCHS = 70
model = AutoModelForQuestionAnswering.from_pretrained("bert-base-uncased")
model.to(device)


from peft import get_peft_model, AdaLoraConfig, TaskType, PeftModel  # Import AdaloraConfig

# Set up the Adalora config (Assuming AdaloraConfig behaves similarly to LoraConfig)
peft_config = AdaLoraConfig(
    task_type=TaskType.QUESTION_ANS,
    inference_mode=False,
    r=8,  # matrix rank
    lora_alpha=16,  # scaling factor
    lora_dropout=0.4,
)


# Wrap model
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

# Optimizer and scheduler
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
num_training_steps = EPOCHS * len(train_dataloader)
scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=len(train_dataloader) * 0.5,  # warmup = half an epoch
    num_training_steps=num_training_steps,
)

# Training loop
train_losses = []
val_losses = []
MAX_GRAD_NORM = 1.0  # Gradient clipping threshold

# Training function
def train(model, train_dataloader, val_dataloader, device, optimizer, scheduler):
    model.train()  # Set model to training mode
    train_losses = []  # To store training losses
    val_losses = []  # To store validation losses

    for epoch in range(EPOCHS):
        epoch_loss = []  # To store average loss for the epoch
        total_loss = 0  # Accumulated loss for the epoch
        print(f"Starting Epoch {epoch + 1}/{EPOCHS}")

        for step, batch in enumerate(train_dataloader):
            start = time.time()

            # Move batch to device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            start_positions = batch["start_positions"].to(device)
            end_positions = batch["end_positions"].to(device)

            # Zero gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(
                input_ids,
                attention_mask=attention_mask,
                start_positions=start_positions,
                end_positions=end_positions,
            )

            # Compute loss
            loss = outputs.loss
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"NaN or Inf loss detected at step {step}. Skipping this batch.")
                continue

            # Backward pass
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=MAX_GRAD_NORM)

            # Update weights
            optimizer.step()
            scheduler.step()

            # Accumulate loss
            total_loss += loss.item()

            # Log training progress
            if step % 400 == 0:
                train_losses.append(loss.item())

                # Validation step
                val_batch = next(iter(val_dataloader))
                with torch.no_grad():
                    model.eval()  # Set model to evaluation mode
                    val_outputs = model(
                        val_batch["input_ids"].to(device),
                        attention_mask=val_batch["attention_mask"].to(device),
                        start_positions=val_batch["start_positions"].to(device),
                        end_positions=val_batch["end_positions"].to(device),
                    )
                    val_loss = val_outputs.loss
                    val_losses.append(val_loss.item())
                model.train()  # Set model back to training mode

                # Print progress
                print(
                    f"Epoch: {epoch + 1}/{EPOCHS} | "
                    f"Step: {step}/{len(train_dataloader)} | "
                    f"Train Loss: {loss.item():.5f} | "
                    f"Val Loss: {val_loss.item():.5f} | "
                    f"LR: {scheduler.get_last_lr()[0]:.5f} | "
                    f"Time per Batch: {time.time() - start:.2f}s"
                )

        # Calculate average loss for the epoch
        avg_train_loss = total_loss / len(train_dataloader)
        epoch_loss.append(avg_train_loss)
        print(f"Epoch {epoch + 1} Average Train Loss: {avg_train_loss:.5f}")


train(model, train_dataloader, val_dataloader, device, optimizer, scheduler)
model.save_pretrained("./saved_model_adalora_BERT")

# Plotting and evaluation code remains the same
plt.figure(figsize=(12, 8))
plt.title("Training Loss")
plt.plot(train_losses, label="Train")
plt.xlabel("Step")
plt.ylabel("Loss")
plt.xticks(
    ticks=plt.xticks()[0][1:], labels=400 * np.array(plt.xticks()[0][1:], dtype=int)
)
plt.legend()
plt.show()
plt.savefig('adalora_BERT.png')

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

  Inputs;
    -batch: The data batch returned from the DataLoader
    -output: Output of the model when given `batch`
    -inference (bool=False): Indicates if labels are available and decodes + returns them if so
  Returns:
    -2-tuple of numpy arrays of length `batch_size` indicating the model predictions and the ground truth labels respectively
    -Note: If set to inference mode (i.e. no labels), only the predictions are returned, and not in 1-tuple form.
  """

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
      batch["sequence_ids"],
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

  # Fix whitespaces, convert lowercase
  text = " ".join(text.split()).lower()

  # Remove punctuation
  text = text.translate(str.maketrans("", "", string.punctuation))

  return text

def exact_match(preds, labels):
  """
  Calculates the exact match score between predictions and labels.
  Normalizes the predictions and labels first, then computes the proportion of equality between normalized predictions and labels.

  Input:
      -preds (np.array): Array of prediction strings
      -labels (np.array): Array of label strings
  Returns:
      -Exact match score (float) between the normalized predictions and labels

  """
  # Normalize predictions and labels
  preds = np.vectorize(normalization)(preds)
  labels = np.vectorize(normalization)(labels)

  return np.mean(preds == labels)





def f1(preds, labels):

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



# Prediction function remains the same
def predict(model, tokenizer, question, context):
    input = tokenizer(
        question,
        context,
        max_length=384,
        truncation="only_second",
        padding="max_length",
    )

    for key in input:
        input[key] = torch.tensor(input[key], dtype=torch.int64).unsqueeze(0)

    input["sequence_ids"] = torch.tensor(
        np.array(input.sequence_ids(), dtype=float)
    ).unsqueeze(0)

    model.eval()
    with torch.no_grad():
        inference_output = model(
            input["input_ids"].to(device),
            attention_mask=input["attention_mask"].to(device),
        )
    pred = postprocess(input, inference_output, inference=True)[0]

    return pred
def eval_acc(model, val_dataloader):
    # Setting up the evaluation metrics
    meteor = evaluate.load("meteor")
    rouge = evaluate.load("rouge")
    bleu = evaluate.load("bleu")
    cross_encoder = CrossEncoder("cross-encoder/stsb-roberta-large")

    # Put the model into evaluation mode
    model.eval()

    # Initialize lists to store accuracy scores
    em_list = []
    f1_list = []
    meteor_list = []
    rouge_1_list, rouge_2_list, rouge_L_list = [], [], []
    bleu_list = []
    sas_list = []

    # Evaluate model
    with torch.no_grad():
        print("Evaluating Validation Accuracies:")
        for batch in tqdm(val_dataloader):
            output = model(
                batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                start_positions=batch["start_positions"].to(device),
                end_positions=batch["end_positions"].to(device),
            )

            preds, labels = postprocess(
                batch, output, inference=False, n_best=10, max_answer_len=30
            )

            # Filter out empty predictions
            valid_indices = [i for i, pred in enumerate(preds) if pred.strip()]
            if not valid_indices:
                continue  # Skip this batch if all predictions are empty

            preds = np.array([preds[i] for i in valid_indices])
            labels = np.array([labels[i] for i in valid_indices])

            # Compute accuracy
            em_val = exact_match(preds, labels)
            f1_val = f1(preds, labels)
            meteor_val = meteor.compute(predictions=preds, references=labels)["meteor"]

            rouge_val = rouge.compute(predictions=preds, references=labels)
            rouge_1 = rouge_val["rouge1"]
            rouge_2 = rouge_val["rouge2"]
            rouge_L = rouge_val["rougeL"]

            bleu_val = bleu.compute(predictions=preds, references=labels)["bleu"]
            sas_val = sas(cross_encoder, preds, labels)

            # Append accuracy scores to the corresponding lists
            em_list.append(em_val)
            f1_list.append(f1_val)
            meteor_list.append(meteor_val)
            rouge_1_list.append(rouge_1)
            rouge_2_list.append(rouge_2)
            rouge_L_list.append(rouge_L)
            bleu_list.append(bleu_val)
            sas_list.append(sas_val)

    # Compute and print average accuracy scores
    em_score = np.mean(em_list)
    f1_score = np.mean(f1_list)
    meteor_score = np.mean(meteor_list)
    rouge_1_score = np.mean(rouge_1_list)
    rouge_2_score = np.mean(rouge_2_list)
    rouge_L_score = np.mean(rouge_L_list)
    bleu_score = np.mean(bleu_list)
    sas_score = np.mean(sas_list)

    print(f"\n\nExact Match: {em_score}")
    print(f"F1: {f1_score}")
    print(f"METEOR: {meteor_score}")
    print(f"ROUGE-1: {rouge_1_score}")
    print(f"ROUGE-2: {rouge_2_score}")
    print(f"ROUGE-L: {rouge_L_score}")
    print(f"BLEU: {bleu_score}")
    print(f"SAS: {sas_score}")

    metrics_dict = {
        "EM": em_score,
        "F1": f1_score,
        "METEOR": meteor_score,
        "ROUGE-1": rouge_1_score,
        "ROUGE-2": rouge_2_score,
        "ROUGE-L": rouge_L_score,
        "BLEU": bleu_score,
        "SAS": sas_score,
    }

    return metrics_dict

# Evaluation and prediction code remains the same
metrics_dict = eval_acc(model, val_dataloader)
df_metrics = pd.DataFrame([metrics_dict])
df_metrics.to_csv('adalora_BERT.csv', index=False)
print(df_metrics)

# Load in our fine-tuned model
config = AdaLoraConfig.from_pretrained("./saved_model_adalora_BERT")
model = AutoModelForQuestionAnswering.from_pretrained(config.base_model_name_or_path)
model = PeftModel.from_pretrained(model, "./saved_model_adalora_BERT")
tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path, use_fast=True)
model.to(device)








context = "When combining orders into one control batch using QS61, it is essential to review the relevant SAP Help documentation for instructions and familiarize yourself with material specification documentation to ensure accurate information. Utilizing SAP's reporting features can also be helpful in generating detailed reports on material specifications and descriptions. Additionally, proper configuration of orders for combination in QS61 is crucial, as well as reviewing and validating the combined control batch for accuracy and completeness. To achieve this, it is recommended to implement robust validation checks to ensure data accuracy and completeness, and leveraging SAP's support resources to resolve any issues or queries related  to QS61 functionality. Moreover, it is also a good practice to utilize QS61 as a tool to consolidate and manage multiple orders, while ensuring that all material specifications and descriptions are accurately captured. By following these best practices, users can ensure a successful combination of orders into one control batch using QS61."


question = "QS61 how it works"
pred = predict(model, tokenizer, question, context)
print(pred)