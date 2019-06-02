import os
import sys
sys.path.append(os.path.abspath('../'))

import torch
from BERT import BertConfig, BertTokenizer, BertForQuestionAnswering
from run_MRC import SquadExample, convert_examples_to_features, run_evaluate

def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()

model = BertForQuestionAnswering.from_pretrained("../../Pretrained/ERNIE/")
for name, param in model.named_parameters():
    if param.requires_grad:
        print(name)
model.eval()

input_ids = torch.LongTensor([[31, 11, 51, 99, 101], [15, 5, 0, 1, 2]])
start_logits, end_logits = model(input_ids)
print("start_logits: ", start_logits, " end_logits: ", end_logits)

vocab_file = "../../Pretrained/ERNIE/vocab.txt"
tokenizer = BertTokenizer.from_pretrained(vocab_file, do_lower_case=True)
model_state_dict = torch.load("../../Model/MRC/pytorch_model.bin", map_location="cpu")
model = BertForQuestionAnswering.from_pretrained("../../Pretrained/ERNIE/", state_dict=model_state_dict)
model.eval()

query = "Linux的作者是谁？"
passage = "托瓦兹利用个人时间及器材创造出了Linux这套系统。"
