import torch
from BERT import BertConfig, BertForQuestionAnswering

model = BertForQuestionAnswering.from_pretrained("../Pretrained/ERNIE/")
for name, param in model.named_parameters():
    if param.requires_grad:
        print(name)
model.eval()

input_ids = torch.LongTensor([[31, 11, 51, 99, 101], [15, 5, 0, 1, 2]])
start_logits, end_logits = model(input_ids)
print("start_logits: ", start_logits, " end_logits: ", end_logits)