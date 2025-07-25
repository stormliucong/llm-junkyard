import torch
import torch.optim as optim
import torch.nn as nn
import tiktoken
from miniGPTDataset import train_datasetloader, val_datasetloader, SameSeqDataset, SimpleLetterTokenizer
from miniTransformer import Transformer
from tqdm import tqdm
import gc
import os
from torch.utils.data import Subset



class Trainer:
    def __init__(self, model, device, train_loader, val_loader, learning_rate=1e-1):
        self.model = model
        self.device = device
        self.model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss()
        self.best_val_loss = float('inf')
        self.epoch = 0

    def train(self, epochs):
        for epoch in range(epochs):
            self.model.train()
            progress_bar = tqdm(self.train_loader, desc=f"Training Epoch {epoch+1}/{epochs}", unit="batch")
            for batch_idx, batch in enumerate(progress_bar):
                progress_bar.set_description(f"Training Epoch {epoch+1}/{epochs} - Batch {batch_idx+1}/{len(self.train_loader)}")
                inputs, outputs, targets = batch
                inputs, outputs, targets = inputs.to(self.device), outputs.to(self.device), targets.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(inputs, outputs) # dimensions: (batch_size, seq_length, vocab_size)
                loss = self.criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1))
                loss.backward()
                self.optimizer.step()
                # Update only every 100 steps
                if (batch_idx + 1) % 100 == 0 or (batch_idx + 1) == len(self.train_loader):
                    
                    progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
                    progress_bar.update(100)
                if self.device.type == 'cuda':
                    torch.cuda.empty_cache()
                elif self.device.type == 'mps':
                    gc.collect()
                    
            self.epoch += 1
            self.validate()

    def validate(self):
        self.model.eval()
        with torch.no_grad():
            progress_bar = tqdm(self.val_loader, desc="Validating", unit="batch")
            for batch_idx, batch in enumerate(progress_bar):
                progress_bar.set_description(f"Validating - Batch {batch_idx+1}/{len(self.val_loader)}")
                inputs, outputs, targets = batch
                inputs, outputs, targets = inputs.to(self.device), outputs.to(self.device), targets.to(self.device)                # forward pass
                outputs = self.model(inputs, output_ids=targets)
                loss = self.criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1))
                progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
                progress_bar.update(1)
                if loss.item() < self.best_val_loss:
                    self.best_val_loss = loss.item()
                    self.save_model('best_model.pth')
                # collect gc
                if self.device.type == 'cuda':
                    torch.cuda.empty_cache()
                elif self.device.type == 'mps':
                    gc.collect()
    
    def save_model(self, path):
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epoch': self.epoch
        }
        torch.save(checkpoint, path)
        print(f"Model saved to {path}")
        
        
    def load_model(self, path):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epoch = checkpoint['epoch']
        self.model.to(self.device)
        print(f"Model loaded from {path}")

if __name__ == "__main__":
    # Example usage
    
    batch_size = 128
    max_seq = 4
    d_model = 4
    d_ff = 4
    n_blocks = 1
    n_heads = 1
    drop_out_rate = 0.1
    learning_rate = 1e-3
    epochs = 10
    v_size = SimpleLetterTokenizer().n_vocab
    print(f"Vocabulary size: {v_size}")
    start_token_id = v_size + 1
    end_token_id = v_size + 2
    v_size = v_size + 2
    print(f"Start token ID: {start_token_id}, End token ID: {end_token_id}")
    print(f"Using vocabulary size: {v_size} (including start and end tokens)")
    # Mac M1 chip
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"load dataset with max_seq: {max_seq}, start_token_id: {start_token_id}, end_token_id: {end_token_id}")
    train_dataset = SameSeqDataset(max_seq=max_seq, start_token_id=start_token_id, end_token_id=end_token_id, train=True)
    val_dataset = SameSeqDataset(max_seq=max_seq, start_token_id=start_token_id, end_token_id=end_token_id, train=False)
    # dataset = Subset(dataset, [0])  # Use every 10th sample for faster training
    train_loader = train_datasetloader(train_dataset, batch_size=batch_size)
    val_loader = val_datasetloader(val_dataset, batch_size=batch_size)
    print(f"load model with v_size: {v_size}, max_seq: {max_seq}, d_model: {d_model}, drop_out_rate: {drop_out_rate}, d_ff: {d_ff}, n_blocks: {n_blocks}, n_heads: {n_heads}")
    print(f"Number of training samples: {len(train_loader.dataset)}")
    print(f"Number of validation samples: {len(val_loader.dataset)}")
    # if best model exists, load it
    model_path = 'best_model.pth'
    if os.path.exists(model_path):
        print(f"Loading model from {model_path}")
        model = Transformer(v_size=v_size, max_seq=max_seq, d_model=d_model, drop_out_rate=drop_out_rate, d_ff=d_ff, n_blocks=n_blocks, n_heads=n_heads, pad_idx=0)
        trainer = Trainer(model, device, train_loader, val_loader, learning_rate)
        trainer.load_model(model_path)
    else:
        print(f"No pre-trained model found. Initializing a new model.")
        # Initialize a new model
        model = Transformer(v_size=v_size, max_seq=max_seq, d_model=d_model, drop_out_rate=drop_out_rate, d_ff=d_ff, n_blocks=n_blocks, n_heads=n_heads, pad_idx=0)
        print(f"load trainer with model: {model}, device: {device}")
        trainer = Trainer(model, device, train_loader, val_loader, learning_rate)
    print(f"Start training for {epochs} epochs")
    trainer.train(epochs=epochs)
    print(f"Training completed. Best validation loss: {trainer.best_val_loss:.4f}")
    
    # test an example
    input_seq = "ABC"
    input_ids = SimpleLetterTokenizer().encode(input_seq)
    input_ids = input_ids + [0]
    print(f"Start token ID: {start_token_id}, End token ID: {end_token_id}")
    # convert to tensor and add batch dimension
    input_ids = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0).to(device)
    print(f"Input sequence: {input_seq}, Input IDs: {input_ids}")
    output_token_ids = model.generate(
        input_ids=input_ids,
        max_length=max_seq,
        start_token_id=start_token_id,
        end_token_id=end_token_id
    )
    
    output_token_ids = output_token_ids[0].tolist()
    # remove start token
    output_token_ids = output_token_ids[1:]  # remove the start token
    # remove end token if it exists
    if end_token_id in output_token_ids:
        output_token_ids = output_token_ids[:output_token_ids.index(end_token_id)]

    output_seq = SimpleLetterTokenizer().decode(output_token_ids)
    print(f"Input sequence: {input_seq}, Generated sequence: {output_seq}")
    