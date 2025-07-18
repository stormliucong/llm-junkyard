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

      input_ids = torch.tensor(self.tokenizer.encode(prompt_tokens), dtype=torch.long, device=self.device).unsqueeze(0) # add batch dimension
      kv_cache = None
      # calculate output
      for _ in range(self.max_seq):
        if kv_cache is not None:
          kv_cache = kv_cache.to(self.device)
        # forward pass through the model
        if hasattr(self.model, 'forward_with_kv_cache'):
          output, kv_cache = self.model.forward_with_kv_cache(input_ids, kv_cache=kv_cache)
        else:
          # If the model does not support kv_cache, just forward normally
          output = self.model(input_ids[:, -1, :])
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
  
          sort_mask = cum_sum > self.top_p
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
        finished_beams = []
        active_beams = []
        # active beams and finish beams 
        for beam in new_beams:
          input_ids, score = beam
          # Check if the next token is EOS
          if input_ids[0, -1].item() == self.EOS:
            # If EOS, just continue to the next beam
            finished_beams.append(beam)
            continue
          else:
            active_beams.append(beam)
            
        # Add finished beams to candidates
        candidates.extend(finished_beams)
            
        if active_beams:
          # Process active beams
          
          # batch input_ids
          input_ids_batch = torch.cat([beam[0] for beam in active_beams], dim=0)  # b, s
          scores_batch = torch.tensor([beam[1] for beam in active_beams], dtype=torch.float, device=self.device)  # b
          
          # forward pass through the model
          output = self.model(input_ids_batch)  # b, s, v
          
          # Get the logits and apply temperature
          logits = output[:, -1, :] / self.temperature # b, v
          
          # Sample from top n tokens
          log_prob = F.log_softmax(logits, dim=-1) # b, v
          top_k_log_prob, top_k_idx = torch.topk(log_prob, self.top_n, dim=-1) # b, top_n
          
          # Create new beam with 
          scores_batch = scores_batch.unsqueeze(1) + top_k_log_prob  # b, top_n
          candidate_next_input_id_batch = top_k_idx.unsqueeze(2)  # b, top_n, 1
          # Expand is a view operation, so it does not allocate new memory
          candidate_beam_batch = torch.cat([input_ids_batch.unsqueeze(1).expand(-1, self.top_n, -1), candidate_next_input_id_batch], dim=-1)  # b, top_n, s+1
  
          # Flatten the candidates
          candidate_beam_batch = candidate_beam_batch.view(-1, candidate_beam_batch.size(2))  # (b * top_n, s+1)
          scores_batch = scores_batch.view(-1)  # (b * top_n)
          
          # Keep top k beams
          top_k_scores, top_k_indices = torch.topk(scores_batch, self.beam_size, descending=True)  # Get top k scores and indices
          candidate_beam_batch = candidate_beam_batch[top_k_indices]  # Select top k beams based on indices (beam_size, s+1)
          
          # Create new candidates
          for i in range(candidate_beam_batch.size(0)):
            new_beam = (candidate_beam_batch[i].unsqueeze(0), top_k_scores[i].item())
            candidates.append(new_beam)
        # Update new beams for the next iteration
        new_beams = candidates
        
        # check if all beams are EOS
        if all(beam[0][0, -1].item() == self.EOS for beam in candidates):
          break
      # decode the generated tokens by selecting the best beam
      best_input_ids = candidates[0][0]  # Select the best beam 
      # Decode the generated tokens
      generated_tokens = self.tokenizer.decode(best_input_ids[0].cpu().tolist())
    return generated_tokens
        
        
        
      
      

    

    
    
    
    
    
