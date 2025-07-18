import torch
import torch.nn.functional as F


class GPTGenerator():
  def __init__(self, model, tokenizer, device, EOS = 0, max_seq = 100, top_n = 5, top_p = 0.8, temperature = 0.8, beam_size = 5):
    self.model = model
    self.device = device
    self.model.to(device)
    self.tokenizer = tokenizer
    self.max_seq = max_seq
    self.top_n = top_n
    self.top_p = top_p
    self.temperature = temperature
    self.EOS = EOS
    self.beam_size = beam_size

  def generate_tokens(self,prompt_tokens):

    self.model.eval()
    with torch.no_grad():
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

      input_ids.squeeze
      # decode next code
      generate_tokens = self.tokenizer.decode(input_ids[0].cpu().tolist())
    return generate_tokens
  
  def generate_tokens_beams(self,prompt_tokens):
    self.model.eval()
    with torch.no_grad():
      assert isinstance(prompt_tokens, str), "prompt tokens should be a string"
      
      input_ids = torch.tensor(self.tokenizer.encode(prompt_tokens), dtype=torch.long, device=self.device).unsqueeze(0) # add batch dimension
      # Initialize beams
      new_beams = [(input_ids, 0)]  # (input_ids, score)
      for _ in range(self.max_seq):
        candidates = []
        for beam in new_beams:
          input_ids, score = beam
          # Check if the next token is EOS
          if input_ids[0, -1].item() == self.EOS:
            # If EOS, just continue to the next beam
            candidates.append(beam)
            continue
          output = self.model(input_ids)
          # Get the logits and apply temperature
          logits = output[:, -1, :] / self.temperature # b, v
          # Sample from top n tokens
          log_prob = F.log_softmax(logits, dim=-1) # b, v
          top_k_log_prob, top_k_idx = torch.topk(log_prob, self.top_n, dim=-1) # b, top_n

          for i in range(self.top_n):
            k_idx = top_k_idx[0, i]
            # Create new beam with the next token
            candidate_next_token = k_idx.unsqueeze(0).unsqueeze(0)  # Reshape to (1, 1)
            candidate_score = score + top_k_log_prob[0, i].item()
            # Create new beam
            new_beam = (torch.cat([input_ids, candidate_next_token], dim=-1), candidate_score)
            candidates.append(new_beam)
        # Keep top k beams
        candidates = sorted(candidates, key=lambda x: x[1], reverse=True)[:self.beam_size]
        
        # check if all beams are EOS
        if all(beam[0][0, -1].item() == self.EOS for beam in candidates):
          break
      # decode the generated tokens by selecting the best beam
      input_ids = candidates[0][0]  # Select the best beam 
      # Decode the generated tokens
      generated_tokens = self.tokenizer.decode(input_ids[0].cpu().tolist())
    return generated_tokens
        
        
        
      
      

    

    
    
    
    
    
