import numpy as np
import torch
from BERT import BertConfig, BertTokenizer, BertForSequenceClassification

id2relation = []
with open("../Data/relation2id.txt", "r", encoding="utf8") as f:
    for line in f:
        relation = line.strip().split(" ")[0]
        id2relation.append(relation)

print("id2relation: ", id2relation)

def _truncate_seq_pair_RE(tokens_name_1, tokens_name_2, tokens_psg, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_name_1) + len(tokens_name_2) + len(tokens_psg) 
        if total_length <= max_length:
            break
        tokens_psg.pop()

vocab_file = "../Chinese/vocab.txt"
tokenizer = BertTokenizer.from_pretrained(vocab_file, do_lower_case=True)

model_state_dict = torch.load("../Model/BERT_RE.bin", map_location="cpu")
model = BertForSequenceClassification.from_pretrained("../Chinese/", state_dict=model_state_dict, num_labels=12)
model.eval()

input_name_1 = "吕惠如"
input_name_2 = "吕美荪"
#input_psg = "王大牛和李晓华为谁是论文的第一作者争得头破血流。"
#input_psg = "对于贾政来说，最头疼的就是混蛋儿子贾宝玉的婚事。"
#input_name_2 = "贾母"
#input_psg = "项羽的祖父项燕是楚国名将，在秦灭楚的战争中阵亡，其祖先项氏多人也是楚国将领。"
input_psg = "吕惠如任南京女子师范学校校长，吕美荪任奉天女子师范学校校长，吕碧城任天津女子师范学校校长，吕坤秀任厦门女子师范学校教师，姐妹四人，同事教育工作。"

tokens_name_1 = tokenizer.tokenize(input_name_1)
tokens_name_2 = tokenizer.tokenize(input_name_2)
tokens_psg = tokenizer.tokenize(input_psg)

max_seq_length = 128
_truncate_seq_pair_RE(tokens_name_1, tokens_name_2, tokens_psg, max_seq_length - 4)

tokens = ["[CLS]"] + tokens_name_1 + ["[SEP]"] + tokens_name_2 + ["[SEP]"]
segment_ids = [0] * len(tokens)

tokens += tokens_psg + ["[SEP]"]
segment_ids += [1] * (len(tokens_psg) + 1)

input_ids = tokenizer.convert_tokens_to_ids(tokens)

# The mask has 1 for real tokens and 0 for padding tokens. Only real
# tokens are attended to.
input_mask = [1] * len(input_ids)

# Zero-pad up to the sequence length.
padding = [0] * (max_seq_length - len(input_ids))
input_ids += padding
input_mask += padding
segment_ids += padding

input_ids = torch.tensor([input_ids], dtype=torch.long)
input_mask = torch.tensor([input_mask], dtype=torch.long)
segment_ids = torch.tensor([segment_ids], dtype=torch.long)

logits = model(input_ids, segment_ids, input_mask)
#print("logits: ", logits)
logits = logits.detach().cpu().numpy()
outputs = np.argmax(logits, axis=1)
#print("outputs: ", outputs)
print("Relation: ", id2relation[outputs[0]])