import torch
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm


class Trainer:
  def __init__(self, model, device, learning_rate = 1e-3):
    self.device = device
    self.model = model
    self.model.to(self.device)
    self.optim = optim.AdamW(self.model.parameters(), learning_rate)
    self.ce = nn.CrossEntropyLoss()

  def train_epoch(self, train_dataloader):
    self.model.train()
    
    progress_bar = tqdm(train_dataloader, "Trainning")
    total_loss = 0
    batch_n = 0
    
    for batch_id, (input_ids, target) in enumerate(progress_bar):

      # Move to device
      input_ids = input_ids.to(self.device)
      target = target.to(self.device) # b, s

      # Zero gradient
      self.optim.zero_grad()

      # Forward Pass
      logits = self.model(input_ids) # b, s, v
      

      # Calculate Loss
      loss = self.ce(logits.view(-1, logits.size(-1)) , target.view(-1))

      # Backward pass
      loss.backward()

      # Model update
      self.optim.step()

      # set progress bar
      total_loss += loss.item()
      batch_n += 1
      progress_bar.set_postfix(f"loss: {loss.item():4f}")
    return total_loss / batch_n

  def val(self,val_dataloader):
    self.model.eval()
    with torch.no_grad():
      total_loss = 0
      batch_n = 0
      progress_bar = tqdm(val_dataloader, "Validating")
      
      for batch_id, (input_ids, target) in enumerate(progress_bar):
        # Move to device
        input_ids = input_ids.to(self.device)
        target = target.to(self.device) # b, s
        
        # forward pass
        logits = self.model(input_ids, target)
        
        # cal loss 
        loss = self.ce(logits.view(-1, logits.size(-1)) , target.view(-1))

        # set progress bar
        total_loss += loss.item()
        batch_n += 1
        progress_bar.set_postfix(f"loss: {loss.item():4f}")
    return total_loss / batch_n

  def train(self, train_dataloader, val_dataloader, epoch_size):
      
      for epochs in tqdm(range(epoch_size), "Epochs"):
        
        average_train_loss = self.train_epoch(train_dataloader)
        print(f"total_loss in training in this epoch is ({average_train_loss:4f})")
        average_val_loss = self.val(val_dataloader)
        print(f"total_loss in val in this epoch is ({average_val_loss:4f})")

  def save_checkpoint(self, file_path):
    checkpoint = {
      "model": self.model.state_dict(),
      "optim": self.optim.state_dict()
    }
    torch.save(checkpoint, file_path)
    return None

  def load_checkpoint(self,file_path):
    checkpoint = torch.load(file_path)
    self.model.load_state_dict(checkpoint['model'])
    self.optim.load_state_dict(checkpoint['optim'])
    return None
        
      
      
      
      
    
