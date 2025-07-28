import torch
from torch.utils.data import Dataset, DataLoader
import tiktoken
import requests
import os
import random
import string


TRAINING_DATASET = "trainset.txt"
VAL_DATASET = "val.txt"
BATCH_SIZE = 4

class GPTDataset(Dataset):
  def __init__(self, file_path, max_seq=512):
    self.max_seq = max_seq
    with open(file_path,"r") as f:
      content = f.read()

    # tokenize and encoder
    self.tokenizer = tiktoken.get_encoding("GPT2")
    self.v_size = self.tokenizer.n_vocab
    assert self.v_size > 0, "Tokenizer vocabulary size is zero."
    self.sequences = self.tokenizer.encode(content)
      
    # make sure the boundary is not exceeded
    self.num_seq = len(self.sequences) - self.max_seq - 1
    assert self.num_seq > 0, "Dataset too short for given max_seq"

  def __len__(self):
    return self.num_seq

  def __getitem__(self, idx):
    # each instance with overlap sliding window 1 token.
    input_ids = self.sequences[idx : self.max_seq]
    target = self.sequences[idx + 1 : self.max_seq + 1]
    
    # add padding only occur when num_seq = 0.
    if len(input_ids) < self.max_seq:
      input_ids.extend([0] * (self.max_seq - len(input_ids)))
      target.extend([0] * ( self.max_seq - len(target)))
                       
    return torch.tensor(input_ids, dtype=torch.long), torch.tensor(target, dtype = torch.long)

class SimpleLetterTokenizer:
    def __init__(self):
        self.char2id = {chr(i + 65): i + 1 for i in range(26)}  # A-Z → 1-26
        self.id2char = {i + 1: chr(i + 65) for i in range(26)}
        self.pad_token_id = 0
        self.n_vocab = len(self.char2id) + 1  # +1 for padding token

    def encode(self, text):
        # text must be all uppercase letters
        return [self.char2id[c] for c in text if c in self.char2id]

    def decode(self, ids):
        return ''.join([self.id2char[i] for i in ids if i != self.pad_token_id])


class SameSeqDataset(Dataset):
    # For training encoder-decoder based transformer model.
    def __init__(self, max_seq=512, start_token_id=None, end_token_id=None, train=True):
        self.max_seq = max_seq
        # self.tokenizer = tiktoken.get_encoding("gpt2")
        self.tokenizer = SimpleLetterTokenizer()
        if train:
            content = ''.join(random.choices(string.ascii_uppercase, k=1000000))  # 1 million characters for training
        else:
            content = ''.join(random.choices(string.ascii_uppercase, k=1000))  # 1K characters for validation
        self.sequences = self.tokenizer.encode(content)
        self.num_seq = len(self.sequences) - self.max_seq - 3
        assert self.num_seq > 0, "Dataset too short for given max_seq"
        self.start_token = start_token_id  
        self.end_token = end_token_id 

    def __len__(self):
        return self.num_seq 

    def __getitem__(self, idx):
        
        # input = [1,2,3]
        # output = [-100,3,2,1]
        # target = [3,2,1,-200]
        input_ids = self.sequences[idx : idx + self.max_seq - 1]
        output = [-100] + input_ids[::-1]
        target = input_ids[::-1] + [-200]  # reverse the input_ids for target

        # add padding only occur when num_seq = 0.
        if len(input_ids) < self.max_seq:
            input_ids.extend([0] * (self.max_seq - len(input_ids)))
            output.extend([0] * (self.max_seq - len(output)))
            target.extend([0] * (self.max_seq - len(target)))

        return torch.tensor(input_ids, dtype=torch.long), torch.tensor(output, dtype=torch.long), torch.tensor(target, dtype=torch.long)
      
class ShakespeareGPTDataset(Dataset):
    # For training GPT based transformer model.
    def __init__(self, max_seq=512, start_token_id=None, end_token_id=None, train=True):
        self.max_seq = max_seq
        self.tokenizer = tiktoken.get_encoding("gpt2")
        if os.path.exists("data/t8.shakespeare.txt"):
            # if the file exists, read from it
            with open("data/t8.shakespeare.txt", "r") as f:
                content = f.read()
        else:
            shakespear_content_url = "https://ocw.mit.edu/ans7870/6/6.006/s08/lecturenotes/files/t8.shakespeare.txt"
            # read the content from the URL
            content = requests.get(shakespear_content_url).text
            # save to content to local file
            with open("data/t8.shakespeare.txt", "w") as f:
                f.write(content)
        # content is random sequence of letters for testing.
        if train:
            content = content[:1000000]  # 1 million characters for training
            
        else:
            content = content[1000000:1000000 + 10000]  # 10K characters for validation
        self.sequences = self.tokenizer.encode(content)
        self.num_seq = len(self.sequences) - self.max_seq - 1
        assert self.num_seq > 0, "Dataset too short for given max_seq"
        self.start_token = start_token_id  
        self.end_token = end_token_id 
        
    def __len__(self):
        return self.num_seq 

    def __getitem__(self, idx):
        
        # input = [1,2,3]
        # target = [3,2,1,-200]
        input_ids = self.sequences[idx : idx + self.max_seq - 1]
        target = self.sequences[idx + 1 : idx + self.max_seq]

        # add padding only occur when num_seq = 0.
        if len(input_ids) < self.max_seq:
            input_ids.extend([0] * (self.max_seq - len(input_ids)))
            target.extend([0] * (self.max_seq - len(target)))

        return torch.tensor(input_ids, dtype=torch.long), torch.tensor(target, dtype=torch.long)


def train_datasetloader(dataset, batch_size=BATCH_SIZE):
    return DataLoader(
      dataset = dataset,
      batch_size = batch_size,
      shuffle=True,
      num_workers=8
    )

def val_datasetloader(dataset, batch_size=BATCH_SIZE):
    return DataLoader(
      dataset = dataset,
      batch_size = batch_size,
      shuffle=False,
      num_workers=8
    )

  
  
      

    
