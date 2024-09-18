import warnings

warnings.filterwarnings('ignore')

import torch
from transformers import AutoModel, AutoTokenizer

import chromadb

# phobert = AutoModel.from_pretrained("./models--vinai--phobert-base-v2", local_files_only=True)
#
# tokenizer = AutoTokenizer.from_pretrained("./models--vinai--phobert-base-v2", local_files_only=True)

phobert = AutoModel.from_pretrained("phobert-base-v2")

tokenizer = AutoTokenizer.from_pretrained("phobert-base-v2")

client = chromadb.HttpClient(host="http://127.0.0.1:8000")

bpm_ops_collection = client.get_or_create_collection(name="bpm_ops_test_9")

query_text = "xuất PDF lỗi tách trang"

input_ids = torch.tensor([tokenizer.encode(query_text)])

with torch.no_grad():
    features = phobert(input_ids)

result = ""

distance = 999999

for embedding in features.last_hidden_state.numpy()[0]:
    query_results = bpm_ops_collection.query(
        query_embeddings=[embedding.tolist()],
        n_results=1
    )

    if query_results['distances'][0][0] < distance:
        distance = query_results['distances'][0][0]

        result = query_results['documents'][0][0]

print(distance)

print(result)
