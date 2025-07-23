import json

# Define the output structure
combined_data = []


# Process files from result_1.json to result_10.json
file_name = r"C:\Users\kawka\PycharmProjects\pythonProject\euronlp\testtt\FINAL_CODES\complete_finall.json"

try:
    with open(file_name, 'r', encoding='utf-8') as file:
        data = json.load(file)
        if "data" in data:
            combined_data.extend(data["data"])  # Combine only the "data" field
        else:
            print(f"'data' field not found in {file_name}, skipping.")
except FileNotFoundError:
    print(f"File {file_name} not found, skipping.")

    # Process files from result_1.json to result_10.json
file_name = r"C:\Users\kawka\PycharmProjects\pythonProject\euronlp\testtt\dataset\result\complete_which.json"

try:
    with open(file_name, 'r', encoding='utf-8') as file:
        data = json.load(file)
        if "data" in data:
            combined_data.extend(data["data"])  # Combine only the "data" field
        else:
            print(f"'data' field not found in {file_name}, skipping.")
except FileNotFoundError:
    print(f"File {file_name} not found, skipping.")

# Save the combined "data" into a single JSON file
output_file = "complete_finall.json"
with open(output_file, 'w', encoding='utf-8') as output:
    json.dump({"data": combined_data}, output, indent=4, ensure_ascii=False)

print(f"Combined JSON saved to {output_file}")