import torch.utils.optim as optim
import tqdm


class Trainer:
  def __init__(self, model, device, learning_rate = 1e-3):
    self.train_batch = train_dataloader()
    self.model = model
    self.model.to_device(device)
    self.optim = optim.AdamW(learning_rate)
    self.ce = nn.CrossEntropyLoss()

  def train_epoch(self, train_data, batch_size):
    batch_train_data = self.batch_train_data
    progress_bar = tqdm(batch_train_data, batch_size, "trainining")

    total_loss = 0
    for batch_id, (input_ids, target) in progress_bar:
      batch_size, _ = input_ids.size()
      input_ids.to_device(device)
      target.to_device(device) # b, v
      self.model.train()
      self.optim.zero_grad()

      # Forward Pass
      logits = self.model(input_ids) # b, v

      # Calculate Loss
      loss = self.ce(logits, target)

      # Backward pass
      loss.backward()

      average_loss = loss / batch_size
      total_loss += average_loss
    return total_loss

  def val(self,val_data):
    self.model.eval()
    with torch.zero_grads():
      total_loss = 0
      progress_bar = tqdm(val_data, batch_size, "validating")
      for batch_id, (input_ids, target) in tqdm(progress_bar):
        input_ids.to_device(self.device)
        target.to_device(self.device)
        
        # forward pass
        logits = self.model(input_ids, target)
        # cal loss 
        loss = self.ce(logits, target)
        average_loss = loss / batch_size
      total_loss += average_loss
    return total_loss    

  def train(self, train_data, epoch_size):
      train_data = self.train_data
      val_data = self.val_data
      batch_size, _ = train_data.size()


      for epochs in tqdm(range(n_epochs)):
        batch_train_data = train_data
        total_loss = self.train_epoch(batch_train_data)
        print(f"total_loss in training in this epoch is ({total_loss})")
        batch_val_data = val_data
        total_loss = self.val_epoch(batch_train_data)

  def save_checkpoints(self):
    torch.save(self.mode)
        
        
      
      
      
      
    
