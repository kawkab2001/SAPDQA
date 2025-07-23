import ollama
from typing import Dict, Any, Callable
import re
import json
from sentence_transformers import SentenceTransformer, util
import nltk
from concurrent.futures import ThreadPoolExecutor

class SAPDataProcessor:
    def __init__(self, input_file: str, output_file: str):
        self.input_file = input_file
        self.output_file = output_file
        self.data = None
        self.titles_seen = set()
        self.indices_to_delete = []
        nltk.download('punkt_tab')
        nltk.download('punkt')

    def generate_context(self, question: str, answer: str) -> str:
        prompt = f"Question: {question}\nAnswer: {answer}\nGenerate a formal and resumed and neutral context, avoiding the use of personal pronouns like 'I', 'you', or 'he'. "
        response = ollama.chat(model="llama3.2", messages=[{"role": "user", "content": prompt}])
        context = response['message']['content'] if 'message' in response and 'content' in response['message'] else "No context generated."
        context = re.sub(r'\b(I|you|he|she|we|they|us|them)\b', '', context)
        context = ' '.join(context.split())
        return context

    def summarize_answer(self, answer: str) -> str:
        prompt = f"Answer: {answer}\nSummarize the above answer into a concise version, keeping only the key points and direct answer."
        response = ollama.chat(model="llama3.2", messages=[{"role": "user", "content": prompt}])
        summarized_answer = response['message']['content'] if 'message' in response and 'content' in response['message'] else "No summary generated."
        return summarized_answer

    def is_english(self, text):
        common_english_words = {'the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have', 'i', 'it', 'for', 'not', 'on', 'with'}
        text_words = set(text.split())
        common_words_count = len(text_words.intersection(common_english_words))
        return common_words_count >= 2

    def remove_html_tags(self, text: str) -> str:
        clean_text = re.sub(r'<.*?>', '', text)
        return clean_text



    def find_answer_location_different_context(self, context, answer):
        sentences = nltk.sent_tokenize(context)
        model = SentenceTransformer('all-MiniLM-L6-v2')
        sentence_embeddings = model.encode(sentences, convert_to_tensor=True)
        answer_embedding = model.encode(answer, convert_to_tensor=True)
        similarities = util.pytorch_cos_sim(answer_embedding, sentence_embeddings)
        similarity_threshold = 0.3
        max_similarity = similarities.max()

        if max_similarity < similarity_threshold:
            return None

        best_match_index = similarities.argmax()
        best_sentence = sentences[best_match_index]
        start_index = context.find(best_sentence)
        end_index = start_index + len(best_sentence)
        return start_index, end_index

    def dell(self, text):
        text = text.lower()
        text = re.sub(r'(?<!\w)\'|\'(?!\w)', '', text)
        text = re.sub(r'https?:\/\/\S+', lambda match: match.group(), text)
        text = ''.join(' ' if not c.isalnum() and not c.isspace() and c != "'" else c for c in text)
        return text

    def contains_no_solution(self, answer):
        if answer == "No accepted solution found.":
            return 0
        else:
            return 1
    def process_item(self, idx, item):
        if not any(self.is_english(paragraph['context']) for paragraph in item['paragraphs']):
            self.indices_to_delete.append(idx)

        for paragraph in item['paragraphs']:
            for qa in paragraph['qas']:
                for answer in qa['answers']:
                    if self.contains_no_solution(answer['text']) == 0:
                        self.indices_to_delete.append(idx)

    def summarize_contex(self,answer: str) -> str:
        # Modify the prompt to guide the model to summarize the answer
        prompt = f"context: {answer}\ndelete any things not necessary like punctuation marks and resume it and simplifies it"

        # Use ollama to generate the summarized answer with LLaMA 3.2
        response = ollama.chat(model="llama3.2", messages=[{"role": "user", "content": prompt}])

        # Extract the summarized answer from the response
        summarized_answer = response['message']['content'] if 'message' in response and 'content' in response[
            'message'] else "No summary generated."

        return summarized_answer

    def prepare(self):
        with open(self.input_file, 'r', encoding='utf-8') as f:
            self.data = json.load(f)



        self.indices_to_delete = list(set(self.indices_to_delete))

        for idx in sorted(self.indices_to_delete, reverse=True):
            del self.data['data'][idx]


        for item in self.data['data']:
            if item['title'] not in self.titles_seen:
                print(f"Title: {item['title']}")
                self.titles_seen.add(item['title'])
            for paragraph in item['paragraphs']:
                for qa in paragraph['qas']:
                    for answer in qa['answers']:
                        question = qa['question']
                        qa['question'] = self.remove_html_tags(qa['question'])
                        answer['text'] = self.remove_html_tags(answer['text'])
                        answer['text'] = self.summarize_answer(answer['text'])
                        paragraph['context'] = self.summarize_answer(self.generate_context(question, answer['text']))
                        context = paragraph['context']

                        location = self.find_answer_location_different_context(context, answer['text'])

                        if location is not None:
                            start_index, end_index = location
                            answer['answer_start'] = start_index
                            answer['answer_end'] = end_index
                        else:
                            answer['answer_start'] = 0
                            answer['answer_end'] = len(context)

        with open(self.output_file, 'w', encoding='utf-8') as file:
            json.dump(self.data, file, indent=4)
            print("Finished processing and saved to:", self.output_file)

# Example usage:
for i in range(1,13):
    j=str(i)
    processor = SAPDataProcessor(input_file='result/which_cleaned_test_batch_'+j+'.json', output_file='result/which_result_'+j+'.json')
    processor.prepare()

