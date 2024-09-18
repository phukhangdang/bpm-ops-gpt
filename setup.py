from transformers import AutoModel, AutoTokenizer

phobert = AutoModel.from_pretrained("vinai/phobert-base-v2")

phobert.save_pretrained("phobert-base-v2")

tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base-v2")

tokenizer.save_pretrained("phobert-base-v2")
