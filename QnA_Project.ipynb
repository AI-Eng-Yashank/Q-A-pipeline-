import os
import json
from datasets import load_dataset
from transformers import pipeline
import nltk
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('maxent_ne_chunker_tab')
nltk.download('words')

# Set environment variables
os.environ["OPENAI_API_KEY"] = "fake-key"
os.environ["OPENAI_BASE_URL"] = "http://localhost:11434/v1"
os.environ["OPENAI_MODEL_NAME"] = "llama3.2"

# Debug environment variables
print("OPENAI_API_KEY:", os.getenv("OPENAI_API_KEY", "Not Set"))
print("OPENAI_BASE_URL:", os.getenv("OPENAI_BASE_URL", "Not Set"))
print("OPENAI_MODEL_NAME:", os.getenv("OPENAI_MODEL_NAME", "Not Set"))

# Initialize Hugging Face pipeline for question generation and answering
def initialize_pipeline():
 return pipeline("text2text-generation", model="google/flan-t5-large",
max_length=500)
question_model = initialize_pipeline()
answer_model = initialize_pipeline()

# Dataset Extraction: Load and inspect the dataset
def extract_content(dataset_path):
 try:
 dataset = load_dataset("json", data_files=dataset_path)
 print(f"Loaded dataset: {dataset}")
 extracted = []
 for entry in dataset["train"]:
 if "page" in entry and "content" in entry and entry["content"]
is not None:
 extracted.append({"page": entry["page"], "content":
entry["content"]})
else:
 print(f"Skipping invalid entry: {entry}")
 print(f"Extracted content: {extracted}")
 return extracted
 except Exception as e:
 print(f"Error loading dataset: {e}")
 return []
 
# Question Generation: Generate questions based on content
def generate_questions(content):
 question_prompt = f"Create a question based on the following
text:\n\n{content}"
 try:
 response = question_model(question_prompt)
 print(f"Generated question: {response}")
 return [resp["generated_text"].strip() for resp in response]
 except Exception as e:
 print(f"Error generating question: {e}")
 return []
 
# Answer Generation: Generate answers for the questions
def generate_answers(content, questions):
 answers = []
 try:
 for question in questions:
 prompt = f"Based on the following
content:\n\n{content}\n\nAnswer the question:\n{question}"
 response = answer_model(prompt)
 answers.append(response[0]["generated_text"].strip())
 print(f"Generated answers: {answers}")
 return answers
 except Exception as e:
 print(f"Error generating answers: {e}")
 return []
 
# Post-process answers to improve grammar
def post_process_answers(answers):
 from nltk import word_tokenize, pos_tag, ne_chunk
 from nltk.tree import Tree 
processed_answers = []
 for answer in answers:
 tokens = word_tokenize(answer)
 tagged = pos_tag(tokens)
 chunked = ne_chunk(tagged)
 
 # Basic grammar correction and refinement
 processed_answer = " ".join([word for word, pos in tagged if pos
not in ['DT', 'IN', 'CC', 'TO']])
 processed_answers.append(processed_answer)
 return processed_answers
 
# Combine everything in one step
def run_qna_workflow(dataset_path):
 print("Extracting content from dataset...")
 extracted_data = extract_content(dataset_path)
 if not extracted_data:
 print("No valid content extracted.")
 return []
 qna_data = []
 for entry in extracted_data:
 print(f"Processing entry: {entry}")
 content = entry["content"]
 page = entry["page"]
 print(f"Content: {content}")
 
 # Step 1: Generate questions
 questions = generate_questions(content)
 if not questions:
 print(f"No questions generated for page {page}. Skipping.")
 continue 
 
 # Step 2: Generate answers
 answers = generate_answers(content, questions)
 if not answers:
 print(f"No answers generated for page {page}. Skipping.") 
continue 

 # Step 3: Post-process answers
 processed_answers = post_process_answers(answers) 
 
 # Append Q&A data for this entry
 qna_data.append({
 "page": page,
 "questions": questions,
 "answers": processed_answers
 }) 
 
 # Print final Q&A data to verify it's populated
 print("Final Q&A Data:", json.dumps(qna_data, indent=4))
 return qna_data 
 
# Save results to a JSON file
def save_results(data, output_path):
 if not data:
 print("No data to save.")
 return
 print("Saving data...")
 try:
 with open(output_path, "w", encoding="utf-8") as f:
 json.dump(data, f, indent=4)
 print(f"Data saved to {output_path}")
 except Exception as e:
 print(f"Error saving data: {e}")
# Main workflow
def main(input_file, output_file):
 print("Running Q&A workflow...")
 qna_dataset_with_answers = run_qna_workflow(input_file)
 save_results(qna_dataset_with_answers, output_file)
# Input and Output Paths
input_file = "E:/your/file/path/File1_scraped_data_cleaned.json"
output_file = "E:/your/file/path/generated_qna_with_answers.json" 
# Execute the workflow

if __name__ == "__main__":

 main(input_file, output_file)
