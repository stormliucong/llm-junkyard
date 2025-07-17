import torch
import torch.nn.functional as F


class GPTGenerator():
  def __init__(self, model, tokenizer, device, EOS = 0, max_seq = 100, top_n = 5, top_p = 0.8, temperature = 0.8):
    self.model = model
    self.device = device
    self.model.to(device)
    self.tokenizer = tokenizer
    self.max_seq = max_seq
    self.top_n = top_n
    self.top_k = top_n
    self.temperature = temperature
    self.EOS = EOS

  def generate_tokens(self,prompt_tokens):
    
    assert isinstance(prompt_tokens, str), "prompt tokens should be a string"
    
    input_ids = torch.tensor(self.tokenizer.encode(prompt_tokens), dtyple=torch.long, device=self.device)).unsqueeze(0) # add batch dimension

    # calculate output
    for _ in range(self.max_seq):
      output = self.model(input_ids) # 1, s, v
      logits = output[:, -1, :] / self.temperature # 1, v
  
      # sample from top n tokens
      if self.top_n > 0:
        top_k_value, top_k_idx = torch.topk(logits, self.top_n, dim = -1) # 1, top_n
        # create a -inf mask
        mask = torch.full_like(logits, float('-inf'))
        # scatter back
        logits = mask.scatter(-1,top_k_idx,top_k_value)
  
      # sample from collectively less than p
      if self.top_p > 0:
        sort_value, sort_idx = torch.sort(logits, dim = -1,descending = True)
        cum_sum = torch.cumsum(F.softmax(sort_value, dim = -1), dim = -1)

        sort_mask = cum_sum > top_p
        sort_value[sort_mask] = float('-inf')
        logits = torch.full_like(logits, float('-inf'))
        logits.scatter_(-1, sort_idx,sort_value)

      # sample next token
      prob = F.softmax(logits, dim = -1)
      next_token = torch.multinomial(prob, dim = -1, num_samples = 1) # 1,1

      # check EOS
      if next_token.item() == self.EOS:
        break

      input_ids = torch.cat([input_ids, next_token], dim=-1)

    # decode next code
    return self.tokenizer.decode(input_ids).squeeze().cpu().tolist()



      
        
        
        
      
      

    

    
    
    
    
    
