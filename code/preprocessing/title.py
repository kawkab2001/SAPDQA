import json
import re


def generate_title(paragraph):
    # Extract key terms using regex to find all capitalized words, which are likely to be table names or important terms
    key_terms = re.findall(r'\b[A-Z][A-Za-z0-9_]*\b', paragraph)

    # Remove duplicates and sort terms by length
    unique_terms = sorted(set(key_terms), key=len, reverse=True)

    # Create a dynamic title using relevant key terms
    title = "Developing a Comprehensive Audit Trail for " + " & ".join(unique_terms[:5]) + " Approval History"

    return title

with open(r"C:\Users\kawka\PycharmProjects\pythonProject\euronlp\testtt\FINAL_CODES\complete_copy.json", encoding="utf-8") as infile:
    data = json.load(infile)
for item in data['data']:
    for paragraph in item['paragraphs']:
        for qa in paragraph['qas']:
            for answer in qa['answers']:
                qa['question']=item["title"]
                item["title"]=generate_title(paragraph["context"])
with open(r"C:\Users\kawka\PycharmProjects\pythonProject\euronlp\testtt\FINAL_CODES\complete_copy.json", 'w', encoding='utf-8') as file:
    json.dump(data, file, indent=4)
    print("Finished processing and saved to:", "complete_which.json")


