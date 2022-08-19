import random
from torch.utils.data import Dataset, DataLoader
from transformers import BertModel, BertConfig, BertTokenizer
import numpy as np

class TrainDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        text = self.data[index]

        return text


class CollateFunc(object):
    def __init__(self, tokenizer, max_len=256, q_size=160, dup_rate=0.15):
        self.q = []
        self.q_size = q_size
        self.max_len = max_len
        self.dup_rate = dup_rate
        self.tokenizer = tokenizer

    def word_repetition_normal(self, batch_text):
        dst_text = list()
        for text in batch_text:
            actual_len = len(text.split())
            if actual_len > 2:
                dup_len = random.randint(a=0, b=max(
                    1, int(np.floor(self.dup_rate * actual_len))))
                dup_word_index = random.sample(
                    list(range(1, actual_len)), k=dup_len)

                dup_text = ''
                for index, word in enumerate(text.split()):
                    if index > 0:
                        dup_text += ' ' +word
                    else:
                        dup_text += word
                    if index in dup_word_index:
                        if index > 0:
                            dup_text += " " + word
                        else:
                            dup_text += " " + word
                dst_text.append(dup_text)
            else:
                dst_text.append(text)
        return dst_text


    
    def negative_samples(self, batch_src_text):
        batch_size = len(batch_src_text)
        negative_samples = None
        if len(self.q) > 0:
            negative_samples = self.q[:self.q_size]
            # print("size of negative_samples", len(negative_samples))

        if len(self.q) + batch_size >= self.q_size:
            del self.q[:batch_size]
        self.q.extend(batch_src_text)

        return negative_samples

    def __call__(self, batch_text):
        '''
        input: batch_text: [batch_text,]
        output: batch_src_text, batch_dst_text, batch_neg_text
        '''
        batch_pos_text = self.word_repetition_normal(batch_text)
        batch_neg_text = self.negative_samples(batch_text)
        # print(len(batch_pos_text))

        batch_tokens = self.tokenizer(batch_text, max_length=self.max_len,
                                      truncation=True, padding='max_length', return_tensors='pt')
        batch_pos_tokens = self.tokenizer(batch_pos_text, max_length=self.max_len,
                                          truncation=True, padding='max_length', return_tensors='pt')

        batch_neg_tokens = None
        if batch_neg_text:
            batch_neg_tokens = self.tokenizer(batch_neg_text, max_length=self.max_len,
                                              truncation=True, padding='max_length', return_tensors='pt')

        return batch_tokens, batch_pos_tokens, batch_neg_tokens
