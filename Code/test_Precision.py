import torch
from BERT import BertConfig, BertForSequenceClassification

model_state_dict = torch.load("../Model/Precision/pytorch_model.bin", map_location="cpu")
model = BertForSequenceClassification.from_pretrained("../Model/Precision/", state_dict=model_state_dict, num_labels=2)
model.eval()
input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
scores = model(input_ids)
print("scores: ", scores)

label_ids = torch.tensor([0, 0], dtype=torch.long)
loss = model(input_ids, labels=label_ids)
print("loss: ", loss)
print("loss mean: ", loss.mean())