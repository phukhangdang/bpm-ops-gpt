import warnings
from uuid import uuid4

warnings.filterwarnings('ignore')

import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer

import chromadb

phobert = AutoModel.from_pretrained("phobert-base-v2")

tokenizer = AutoTokenizer.from_pretrained("phobert-base-v2")

qa_dataframe = pd.read_excel('qa.xls')

qa_array = qa_dataframe.to_numpy()

client = chromadb.HttpClient(host="http://127.0.0.1:8000")

bpm_ops_collection = client.get_or_create_collection(name="bpm_ops_test_11")

for qa in qa_array:
    input_ids = torch.tensor([tokenizer.encode(qa[0])])

    with torch.no_grad():
        features = phobert(input_ids)

    bpm_ops_collection.add(
        embeddings=[features.pooler_output.squeeze().cpu().numpy().tolist()],
        documents=[qa[1]],
        ids=[str(uuid4())],
    )
