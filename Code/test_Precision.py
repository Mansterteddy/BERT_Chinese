import torch
from BERT import BertConfig, BertForSequenceClassification

model_state_dict = torch.load("../Model/precision.bin")

model = BertForSequenceClassification.from_pretrained("../Chinese/", state_dict=model_state_dict, num_labels=2)
model.eval()
input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
scores = model(input_ids)
print("scores: ", scores)