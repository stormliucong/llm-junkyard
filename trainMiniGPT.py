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
    
    progress_bar = tqdm(train_dataloader, "trainining")
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
      v_size = logitis.size()[-1]
      loss = self.ce(logits.view(-1, logits.size(-1)) , target.view(-1))

      # Backward pass
      loss.backward()

      # Model update
      self.optim.step()

      # set progress bar
      total_loss += loss
      batch_n += 1
      progress_bar.set_postfix(f"loss: {loss}")
    return total_loss / batch_n

  def val(self,val_dataloader):
    self.model.eval()
    with torch.no_grads():
      total_loss = 0
      batch_n += 1
      progress_bar = tqdm(val_dataloader, "validating")
      
      for batch_id, (input_ids, target) in enumerate(progress_bar):
        # Move to device
        input_ids = input_ids.to(self.device)
        target = target.to(self.device) # b, s
        
        # forward pass
        logits = self.model(input_ids, target)
        
        # cal loss 
        v_size = logitis.size()[-1]
        loss = self.ce(logits.view(-1, logits.size(-1)) , target.view(-1))

        # set progress bar
        total_loss += loss
        batch_n += 1
        progress_bar.set_postfix(f"loss: {loss.item()}")
    return total_loss / batch_n

  def train(self, train_dataloader, val_dataloader, epoch_size):
      
      for epochs in tqdm(range(epoch_size)):
        
        average_train_loss = self.train_epoch(train_dataloader)
        print(f"total_loss in training in this epoch is ({average_train_loss})")
        average_val_loss = self.train_epoch(val_dataloader)
        print(f"total_loss in val in this epoch is ({average_val_loss})")
        
        
      
      
      
      
    
