import torch
from torch.utils.data import Dataset, DataLoader
import tiktoken


TRAINING_DATASET = "trainset.txt"
VAL_DATASET = "val.txt"
BATCH_SIZE = batch_size

class GPTDataset(Dataset):
  def __init__(self, file_path, max_seq=512):
    with open(file_path,"r") as f:
      content = f.read()

    # tokenize and encoder
    sequences = titoken.encoder(content, encoder = "GPT2")

    self.num_seq = len(sequences) // max_seq + 1

  def __len__(self):
    return len(self.num_seq)

  def __getitems__(self, idx):
    # each instance
    seq_idx = idx * max_seq
    input_start = seq_idx
    input_end = min(len(sequences), input_end + max_seq)
    target_start = seq_idx + 1
    target_end = min(len(sequences), target_start + max_seq)
    input_ids = sequences[seq_idx : seq_idx + max_seq]
    target = sequence[seq_idx+1 : seq_idx+ 1 + max_seq]

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
      shuffle=True,
      num_workers=8
    )

  
  
      

    
