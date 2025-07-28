from miniGPTDataset import ShakespeareGPTDataset, train_datasetloader, val_datasetloader
from miniGPT import GPT
import torch
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import gc
import os



class Trainer:
  def __init__(self, model, device, learning_rate = 1e-3, weight_decay = 0.01, warmup_steps = 1000, max_steps = 10000, gradient_accumulation_steps = 1, grad_clip = 1.0, save_dir = "./checkpoints"):
    self.device = device
    self.save_dir = save_dir
    self.grad_clip = grad_clip
    self.gradient_accumulation_steps = gradient_accumulation_steps
    # model, optimizer, scheduler
    self.model = model.to(self.device)
    if device == 'cuda':
      self.model = torch.compile(self.model, mode="reduce-overhead") # Only apply this if model is compute-heavy and stable
      
      
    self.optim = optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    # learning rate scheduler with warmup + cosine decay
    self.scheduler = self._get_scheduler(warmup_steps, max_steps)
    self.ce = nn.CrossEntropyLoss(ignore_index=-100)  # ignore padding index in loss calculation
    
    # training state
    self.epoch = 0
    self.global_step = 0
    self.best_val_loss = float('inf')
    
  def _get_scheduler(self, warmup_steps, max_steps):
    def lr_lambda(current_step):
      if current_step < warmup_steps:
        return float(current_step) / float(max(1, warmup_steps))
      # Cosine decay after warmup
      progress = float(current_step - warmup_steps) / float(max(1, max_steps - warmup_steps))
      return max(0.0, 0.5 * (1.0 + torch.cos(torch.pi * progress)))
    return optim.lr_scheduler.LambdaLR(self.optim, lr_lambda)

  def train_epoch(self, train_dataloader):
    self.model.train()
    
    progress_bar = tqdm(train_dataloader, "Trainning for epoch {}".format(self.epoch), leave=False)
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
      
      # Scale loss to avoid overflow
      loss = loss / self.gradient_accumulation_steps if self.gradient_accumulation_steps is not None else loss

      # Backward pass
      loss.backward()

      # Model update
      if (batch_id + 1) % self.gradient_accumulation_steps == 0:
        # Gradient clipping
        if self.grad_clip is not None:
          nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
          # Step optimizer
          self.optim.step()
          # Update scheduler
          self.scheduler.step()
          self.global_step += 1
        

      # set progress bar
      total_loss += loss.item() * self.gradient_accumulation_steps
      batch_n += 1
      progress_bar.set_postfix(f"loss: {loss.item():4f}, step: {self.global_step}")
    return total_loss / batch_n if batch_n > 0 else 0

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
    return total_loss / batch_n if batch_n > 0 else 0

  def train(self, train_dataloader, val_dataloader, epoch_size):

      for epochs in tqdm(range(epoch_size), "Epochs"):
        self.epoch = epochs

        average_train_loss = self.train_epoch(train_dataloader)
        print(f"total_loss in training in this epoch is ({average_train_loss:4f})")
        average_val_loss = self.val(val_dataloader)
        print(f"total_loss in val in this epoch is ({average_val_loss:4f})")
        if average_val_loss < self.best_val_loss:
          self.best_val_loss = average_val_loss
          print(f"New best model saved with loss: {self.best_val_loss:4f}")
          self.save_checkpoint(os.path.join(self.save_dir, f"best_model_epoch_{self.epoch}.pt"))
          
      
        
  def save_checkpoint(self, file_path):
    checkpoint = {
      "model_state_dict": self.model.state_dict(),
      "optim_state_dict": self.optim.state_dict(),
      "scheduler_state_dict": self.scheduler.state_dict(),
      "epoch": self.epoch,
      "global_step": self.global_step,
      "best_val_loss": self.best_val_loss
    }
    torch.save(checkpoint, file_path)
    return None

  def load_checkpoint(self,file_path):
    checkpoint = torch.load(file_path)
    self.model.load_state_dict(checkpoint["model_state_dict"])
    self.optim.load_state_dict(checkpoint["optim_state_dict"])
    self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    self.epoch = checkpoint["epoch"]
    self.global_step = checkpoint["global_step"]
    self.best_val_loss = checkpoint["best_val_loss"]
    print(f"Checkpoint loaded from {file_path}, epoch: {self.epoch}, global_step: {self.global_step}, best_val_loss: {self.best_val_loss:4f}")
    return None
  
if __name__ == "__main__":
    # mps
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    device = torch.device("cuda" if torch.cuda.is_available() else device)
    print(f"Using device: {device}")
    
    # parameters
    max_seq = 32
    d_model = 128
    n_heads = 2
    n_blocks = 2
    d_ff = 256
    drop_out_rate = 0.1
    abs_pos_emb = True  # Use absolute positional embedding
    weight_tier = True  # Use standard projection, not weight tiering
    # training parameters
    batch_size = 2
    learning_rate = 1e-3
    weight_decay = 0.01
    warmup_steps = 1000
    max_steps = 10000
    gradient_accumulation_steps = 64
    grad_clip = 1.0
    save_dir = "./checkpoints"

    train_dataset = ShakespeareGPTDataset(max_seq=max_seq, train=True)
    val_dataset = ShakespeareGPTDataset(max_seq=max_seq, train=False)

    train_datasetloader = train_datasetloader(train_dataset, batch_size=batch_size)
    val_datasetloader = val_datasetloader(val_dataset, batch_size=batch_size)
    
    model = GPT(
        v_size=train_dataset.tokenizer.n_vocab,
        max_seq=max_seq,
        d_model=d_model,
        drop_out_rate=drop_out_rate,
        d_ff=d_ff,
        n_blocks=n_blocks,
        n_heads=n_heads,
        abs_pos_emb=abs_pos_emb,  # Use absolute positional embedding
        weight_tier=weight_tier  # Use standard projection
    )
    print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    trainer = Trainer(
        model=model,
        device=device,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        warmup_steps=warmup_steps,
        max_steps=max_steps,
        gradient_accumulation_steps=gradient_accumulation_steps,
        grad_clip=grad_clip,
        save_dir=save_dir
    )
    print(f"Training with {len(train_datasetloader.dataset)} training samples and {len(val_datasetloader.dataset)} validation samples")
    print(f"Starting training for {max_steps} steps")
    trainer.train(train_datasetloader, val_datasetloader, epoch_size=10)
    print(f"Training completed. Best validation loss: {trainer.best_val_loss:.4f}")    
        
      
      
      
      
    
