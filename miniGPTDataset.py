import torch
from torch.utils.data import Dataset, DataLoader
import tiktoken


TRAINING_DATASET = "trainset.txt"
VAL_DATASET = "val.txt"
BATCH_SIZE = 32

class GPTDataset(Dataset):
  def __init__(self, file_path, max_seq=512):
    self.max_seq = max_seq
    with open(file_path,"r") as f:
      content = f.read()

    # tokenize and encoder
    self.tokenizer = tiktoken.get_encoding("GPT2")
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

def train_datasetloader():
    dataset = GPTDataset(TRAINING_DATASET)
    return DataLoader(
      dataset = dataset,
      batch_size = BATCH_SIZE,
      shuffle=True,
      num_workers=8
    )

def val_datasetloader():
    dataset = GPTDataset(VAL_DATASET)
    return DataLoader(
      dataset = dataset,
      batch_size = BATCH_SIZE,
      shuffle=False,
      num_workers=8
    )

  
  
      

    
