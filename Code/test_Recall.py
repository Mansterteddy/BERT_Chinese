import torch
from BERT import BertConfig, BertForSequenceRelevance

#model_state_dict = torch.load("../Model/relevance.bin")

# CPU Test
#model = BertForSequenceRelevance.from_pretrained("../Chinese/", state_dict=model_state_dict, device="cpu")
model = BertForSequenceRelevance.from_pretrained("../Pretrained/ERNIE/", device="cpu")
for name, param in model.named_parameters():
    if param.requires_grad:
        print(name)

model.eval()
q_input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
p_input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
loss = model(q_input_ids, p_input_ids, labels=True)
print("loss: ", loss)
scores, q_vec, p_vec = model(q_input_ids, p_input_ids)
print("scores: ", scores)
print("q_vec: ", q_vec)
print("p_vec: ", p_vec)


# GPU Test
#model = BertForSequenceRelevance.from_pretrained("../Chinese/", state_dict=model_state_dict, device="cuda")
model = BertForSequenceRelevance.from_pretrained("../Pretrained/ERNIE/", device="cuda")
model.to("cuda")
model.eval()
q_input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]]).to("cuda")
p_input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]]).to("cuda")
loss = model(q_input_ids, p_input_ids, labels=True)
print("loss: ", loss.detach().cpu().numpy())
scores, q_vec, p_vec = model(q_input_ids, p_input_ids)
print("scores: ", scores.detach().cpu().numpy())