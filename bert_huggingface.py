from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModel
import pandas as pd
import qiskit
from qiskit import QuantumCircuit
import qiskit_ibm_runtime
from qiskit_ibm_runtime import QiskitRuntimeService
import numpy as np

#Huggingface Datasets Used: Financial Phrasebank, Stocktwits, FIQA
#Financial Phrasebank - use for building a sentiment analysis model based on financial text
#StockTwits - use to analyze social sentiment on stock movements in real time
#FiQA - news dataset to build comprehensive model that can track news and social media sentiment

#Load all 3 datasets then convert into pandas dataframes 
financial_phrasebank = load_dataset('financial_phrasebank', 'sentences_50agree')
sentiment140 = load_dataset('sentiment140')
financial_news = load_dataset('dilkasithari-IT/sentiment_analysis_financial_news_data')

fpb_df = pd.DataFrame(financial_phrasebank['train'])
s140_df = pd.DataFrame(sentiment140['train'])
fn_df = pd.DataFrame(financial_news['train'])

# Print column names to identify the correct ones if using different dataset
#print(f"Financial PhraseBank columns: {fpb_df.columns}")
#print(f"Sentiment140 columns: {s140_df.columns}")
#print(f"Sentiment Analysis Financial News columns: {fn_df.columns}")

#Map labels to common structure then concatenate into all 3 single dataframe
s140_df['sentiment'] = s140_df['sentiment'].map({0: 0, 4: 2})
fn_df['sentiment'] = fn_df['sentiment'].map({-1: 0, 0: 1, 1: 2})

fpb_df = fpb_df[['sentence', 'label']]
s140_df = s140_df[['text', 'sentiment']]
fn_df = fn_df[['combined_text', 'sentiment']]

fpb_df.columns = ['text', 'sentiment']
fn_df.columns = ['text', 'sentiment']

combined_df = pd.concat([fpb_df, s140_df, fn_df], ignore_index=True)

#Tokenize combined dataframes using BERT tokenizer
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
def tokenize_function(example_text):
    return tokenizer(example_text['text'], padding="max_length", truncation=True)
combined_dataset = Dataset.from_pandas(combined_df)
tokenized_dataset = combined_dataset.map(tokenize_function, batched=True)

#Split Dataset into training and test sets
train_test_split = tokenized_dataset.train_test_split(test_size=0.2)
train_dataset = train_test_split['train']
test_dataset = train_test_split['test']

bert_model = AutoModel.from_pretrained('bert-base-uncased')
def get_bert_embeddings(input_text):
    inputs = tokenizer(input_text, return_tensors='pt', truncation=True, padding=True)
    outputs = bert_model(**inputs)
    embeddings = outputs.last_hidden_state
    return embeddings

service = QiskitRuntimeService()
n_qubits = 4
backend = service.least_busy(n_qubits)

def quantum_circuit(embeddings, weights):
    quantum_circuit = QuantumCircuit(n_qubits)

    for i in range(n_qubits):
        quantum_circuit.rx(embeddings[i], i)

    for i in range(n_qubits):
        quantum_circuit.ry(weights[i,0], i)
        quantum_circuit.rz(weights[i,1], i)

    quantum_circuit.measure_all()
    return quantum_circuit
