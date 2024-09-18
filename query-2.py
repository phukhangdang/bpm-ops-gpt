import warnings

warnings.filterwarnings('ignore')

import torch
from transformers import AutoModel, AutoTokenizer

import chromadb

phobert = AutoModel.from_pretrained("phobert-base-v2")

tokenizer = AutoTokenizer.from_pretrained("phobert-base-v2")

client = chromadb.HttpClient(host="http://127.0.0.1:8000")

bpm_ops_collection = client.get_or_create_collection(name="bpm_ops_test_11")

query_text = "xuất PDF lỗi tách trang"

input_ids = torch.tensor([tokenizer.encode(query_text)])

with torch.no_grad():
    features = phobert(input_ids)

query_results = bpm_ops_collection.query(
    query_embeddings=[features.pooler_output.squeeze().cpu().numpy().tolist()],
    n_results=5
)

for document in query_results['documents'][0]:
    print(document)

    print('-------')
