#encoding=utf-8
#Check the differences between BERT-Chinese, BERT-Multi-Lingual, ERNIE.

import torch
import torch.autograd as autograd
import torch.nn.functional as F
from BERT import BertTokenizer, BertForMaskedLM

def test(tokenizer, model):
    # Tokenized input
    text = "哈尔滨是黑龙江的省会，国际冰雪文化名城。"
    tokenized_text = tokenizer.tokenize(text)
    tokenized_text[0] = "[MASK]"
    tokenized_text[1] = "[MASK]"
    tokenized_text[2] = "[MASK]"
    tokenized_text.insert(0, "[CLS]")
    tokenized_text.append("[SEP]")
    print("Tokenized_text: ", tokenized_text)

    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    print("indexed_tokens: ", indexed_tokens)
    segments_ids = [0 for i in range(len(tokenized_text))]

    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensor = torch.tensor([segments_ids])

    with torch.no_grad():
        predictions = model(tokens_tensor, segments_tensor)

    topk = 5
    output_score = autograd.Variable(predictions[0, 1])
    softmax_prob = F.softmax(output_score, dim=0)
    topk_prob = torch.topk(softmax_prob, topk)[0].tolist()
    topk_prob_id = torch.topk(softmax_prob, topk)[1].tolist()
    predicted_token = tokenizer.convert_ids_to_tokens(topk_prob_id)
    print("Top 5 Token: ", predicted_token)   

    output_score = autograd.Variable(predictions[0, 2])
    softmax_prob = F.softmax(output_score, dim=0)
    topk_prob = torch.topk(softmax_prob, topk)[0].tolist()
    topk_prob_id = torch.topk(softmax_prob, topk)[1].tolist()
    predicted_token = tokenizer.convert_ids_to_tokens(topk_prob_id)
    print("Top 5 Token: ", predicted_token)   

    output_score = autograd.Variable(predictions[0, 3])
    softmax_prob = F.softmax(output_score, dim=0)
    topk_prob = torch.topk(softmax_prob, topk)[0].tolist()
    topk_prob_id = torch.topk(softmax_prob, topk)[1].tolist()
    predicted_token = tokenizer.convert_ids_to_tokens(topk_prob_id)
    print("Top 5 Token: ", predicted_token)   

if __name__ == "__main__":

    # BERT-Chinese
    # Load pre-trained model tokenizer (vocabulary)
    tokenizer = BertTokenizer.from_pretrained("../Pretrained/BERT/vocab.txt")
    # Load pre-trained model (weights)
    model = BertForMaskedLM.from_pretrained("../Pretrained/BERT/")
    model.eval()
    test(tokenizer, model)

    # BERT-Multi-Lingual
    # Load pre-trained model tokenizer (vocabulary)
    tokenizer = BertTokenizer.from_pretrained("../Pretrained/Multi-Lingual/vocab.txt")
    # Load pre-trained model (weights)
    model = BertForMaskedLM.from_pretrained("../Pretrained/Multi-Lingual/")
    model.eval()
    test(tokenizer, model)

    # ERNIE
    # Load pre-trained model tokenizer (vocabulary)
    tokenizer = BertTokenizer.from_pretrained("../Pretrained/ERNIE/vocab.txt")
    # Load pre-trained model (weights)
    model = BertForMaskedLM.from_pretrained("../Pretrained/ERNIE/")
    model.eval()
    test(tokenizer, model)