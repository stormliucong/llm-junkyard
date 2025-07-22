import torch 
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeaderAttention(nn.Module):
	def __init__(self,d_model, n_heads):
		super().__init__()
		self.w_q = nn.Linear(d_model, d_model)
		self.w_k = nn.Linear(d_model, d_model)
		self.w_v = nn.Linear(d_model, d_model)
		self.h = d_model // n_heads
		self.n = n_heads
		assert n_heads * self.h == d_model, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"

		self.w_o = nn.Linear(d_model, d_model)
		self.__init_weights()

	def __init_weights(self):
		for name, param in self.named_parameters():
			if "weight" in name:
				nn.init.xavier_uniform_(param)
			elif "bias" in name:
				nn.init.zeros_(param)

	def forward(self, x, mask=None):
		b,s,d = x.size()
		h = self.h
		n = self.n
  
		if mask is not None:
			assert mask.dim() == 3, "mask should be a 3D tensor"
			assert mask.size(0) == b, "mask should have the same batch size as input x"
			assert mask.size(1) == s, "mask should have the same sequence length as input x"
			# mask should be square and match the sequence length of input x	
			assert mask.size(1) == mask.size(2), "mask should be square"

		# linear projection
		q = self.w_q(x)
		k = self.w_k(x)
		v = self.w_v(x)

		# split and transpose
		q = q.view(b,s,n,h).transpose(1,2) # b,n,s,h
		k = k.view(b,s,n,h).transpose(1,2)
		v = v.view(b,s,n,h).transpose(1,2)

		# matmul, mask, softmax and qkv
		qk = torch.matmul(q, k.transpose(-1,-2)) / math.sqrt(h) # b,n,s,s

		if mask is not None:
			mask = mask.unsqueeze(1)  # b, 1, s, s
			qk.masked_fill_(mask == 0, -1e9) 
		qk = F.softmax(qk, dim = -1)
		qkv = torch.matmul(qk, v) # b,n,s,h

		# transpose, view, and projection
		qkv = qkv.transpose(1,2).contiguous().view(b,s,d)

		return self.w_o(qkv)

	def forward_with_kv_cache(self, x, kv_cache=None):
		''' Forward pass with key-value cache for efficient inference.'''

		assert x.dim() == 3, "input should be a 3D tensor"
		
		b,s,d = x.size()
		h = self.h
		n = self.n
		


		# linear projection only new tokens
		q = self.w_q(x) # b,s,d
		k = self.w_k(x) # b,s,d
		v = self.w_v(x) # b,s,d

		# split and transpose
		q = q.view(b,s,n,h).transpose(1,2) # b,n,s,h
		k = k.view(b,s,n,h).transpose(1,2) # b,n,s,h
		v = v.view(b,s,n,h).transpose(1,2) # b,n,s,h

		# calculate k and v for new x
		if kv_cache is not None:
			k_cache, v_cache = kv_cache
			assert k_cache.size(0) == b, "k_cache should have the same batch size as input"
			assert k_cache.size(1) == n, "k_cache should have the same number of headers as input"
			assert k_cache.size(3) == h, "k_cache should have the same head size as input"
			assert v_cache.size(0) == b, "v_cache should have the same batch size as input"
			assert v_cache.size(1) == n, "v_cache should have the same number of headers as input"
			assert v_cache.size(3) == h, "v_cache should have the same head size as input"
			k = torch.cat([k_cache, k], dim=2) # b,n,s_cache+s,h
			v = torch.cat([v_cache, v], dim=2) # b,n,s_cache+s,h
		# kv_cache is now updated
		kv_cache = (k, v)

		# matmul, mask, softmax and qkv
		qk = torch.matmul(q, k.transpose(-1,-2)) / math.sqrt(h) # b,n,s,s_cache+s

		qk = F.softmax(qk, dim = -1) # b,n,s,s_cache+s
		qkv = torch.matmul(qk, v) # b,n,s,h

		# transpose, view, and projection
		qkv = qkv.transpose(1,2).contiguous().view(b,s,d) # b,s,d

		# final projection
		return self.w_o(qkv), kv_cache

			
class FeedForward(nn.Module):
	def __init__(self, d_model, d_ff, drop_out_rate):
		super().__init__()
		self.fn1 = nn.Linear(d_model, d_ff)
		self.dropout = nn.Dropout(drop_out_rate)
		self.fn2 = nn.Linear(d_ff, d_model)
		# self.ln = nn.LayerNorm(d_model)
		self.__init_weights()
	def __init_weights(self):
		for name, param in self.named_parameters():
			if "weight" in name:
				nn.init.xavier_uniform_(param)
			elif "bias" in name:
				nn.init.zeros_(param)
	def forward(self, x):
		x = self.fn1(x)
		x = F.relu(x)
		x = self.dropout(x)
		x = self.fn2(x)
		return x

class TransformerBlock(nn.Module):
	def __init__(self,d_model, drop_out_rate, d_ff, n_heads):
		super().__init__()
		self.att = MultiHeaderAttention(d_model, n_heads)
		self.ln1 = nn.LayerNorm(d_model)
		self.ln2 = nn.LayerNorm(d_model)
		self.ff = FeedForward(d_model,d_ff, drop_out_rate)
    
	def forward(self, x, mask=None):
		residue1 = x # pre-layer norm
		x= self.ln1(x)
		x = self.att(x, mask)
		x = x + residue1
  
		residue2 = x
		x = self.ln2(x)
		x = self.ff(x)
		x = x + residue2
		return x

	def forward_with_kv_cache(self, x, kv_cache=None):
		residue1 = x
		x = self.ln1(x)
		x, kv_cache = self.att.forward_with_kv_cache(x, kv_cache)
		x = x + residue1
  
		residue2 = x
		x = self.ln2(x)
		x = self.ff(x)
		x = x + residue2
		return x, kv_cache

class SinCosinePositionalEmbedding(nn.Module):
	def __init__(self, d_model, max_seq_len):
		super().__init__()
		self.d_model = d_model
		self.max_seq_len = max_seq_len
		self.register_buffer("positional_embedding", self._generate_positional_embedding()) # no parameters, just a buffer

	def _generate_positional_embedding(self):
		assert self.d_model % 2 == 0, "d_model must be even for SinCosinePositionalEmbedding"
		position = torch.arange(0, self.max_seq_len).unsqueeze(1)  # (max_seq_len, 1)
		# w_k = 1/10000^(2k/d) = exp(-log(10000) * 2k / d_model)
		w_k = torch.exp(-math.log(10000) * 2 * torch.arange(0, self.d_model // 2, dtype=torch.float32) / self.d_model)
		pos_emb = torch.zeros(self.max_seq_len, self.d_model)
		pos_emb[:, 0::2] = torch.sin(position * w_k)  # even indices sin(pos * 2i)
		pos_emb[:, 1::2] = torch.cos(position * w_k)  # odd indices cos(pos * 2i+1)
		return pos_emb.unsqueeze(0)  # (1, max_seq_len, d_model)

	def forward(self, position_ids):
		return self.positional_embedding[:, :position_ids.size(1), :]

class DecoderOnly(nn.Module):
	def __init__(self, v_size, max_seq, d_model, drop_out_rate, d_ff, n_blocks, n_heads, abs_pos_emb=False, weight_tier=False, mask_id=None):
		super().__init__()
		self.ms = max_seq
		self.emb = nn.Embedding(v_size, d_model)
		self.pos = nn.Embedding(max_seq, d_model) if abs_pos_emb else SinCosinePositionalEmbedding(d_model, max_seq)
		self.blocks = nn.ModuleList([TransformerBlock(d_model, drop_out_rate, d_ff, n_heads) for _ in range(n_blocks)])
		self.ln = nn.LayerNorm(d_model)
		self.mask_id = mask_id
		# final projection
		if weight_tier:
			# use original embedding weights for projection
			self.proj = nn.Linear(d_model, v_size, bias=False)
			self.proj.weight = self.emb.weight

		else:
			# standard projection
			self.proj = nn.Linear(d_model, v_size)

	def forward(self, input_ids):
		b, s = input_ids.size()
		assert s <= self.ms, f"Input sequence length {s} exceeds maximum sequence length {self.ms}"
		token_emb = self.emb(input_ids)
		position_ids = torch.arange(0, s, device = input_ids.device).unsqueeze(0) # 1,
		pos_emb = self.pos(position_ids)
		x = token_emb + pos_emb
		casual_mask = torch.tril(torch.ones(s,s, device=input_ids.device)) # s, s
		mask = (input_ids != self.mask_id).float() # b, s
		mask = mask.unsqueeze(1).expand(b, s, s)  # b, s, s
		mask = mask & casual_mask.unsqueeze(0)  # b, s, s
		for block in self.blocks:
			x = block(x, mask)

		# final layernorm and projection
		logits = self.proj(self.ln(x))
		return logits

	def forward_with_kv_cache(self, input_ids, kv_cache=None):
		
		b, s = input_ids.size()
		assert s <= self.ms, f"Input sequence length {s} exceeds maximum sequence length {self.ms}"
		if kv_cache is not None:
			assert len(kv_cache) == len(self.blocks), "kv_cache should have the same length as blocks"
			pos_start_idx = kv_cache[0][0].size(2) 
			input_ids = input_ids[:, pos_start_idx:] # b, s - new_token_start_idx
			b, s = input_ids.size()
		else:
			kv_cache = [None] * len(self.blocks)
			pos_start_idx = 0
		
   
		token_emb = self.emb(input_ids)
		position_ids = torch.arange(pos_start_idx, pos_start_idx + s, device=input_ids.device).unsqueeze(0)  # 1, s - new_token_start_idx
		pos_emb = self.pos(position_ids)
		x = token_emb + pos_emb

		for i, block in enumerate(self.blocks):
			x, kv_cache[i] = block.forward_with_kv_cache(x, kv_cache[i])
		# final layernorm and projection
		logits = self.proj(self.ln(x))
		return logits, kv_cache

class EncoderOnly(nn.Module):
	def __init__(self, v_size, max_seq, d_model, drop_out_rate, d_ff, n_blocks, n_heads, mask_id=None):
		super().__init__()
		self.ms = max_seq
		self.emb = nn.Embedding(v_size, d_model)
		self.pos = nn.Embedding(max_seq, d_model)
		self.blocks = nn.ModuleList([TransformerBlock(d_model, drop_out_rate, d_ff, n_heads) for _ in range(n_blocks)])
		self.ln = nn.LayerNorm(d_model)
		self.proj = nn.Linear(d_model, v_size)
		self.mask_id = mask_id
  
	def forward(self, input_ids):
		b, s = input_ids.size()
		assert s <= self.ms, f"Input sequence length {s} exceeds maximum sequence length {self.ms}"
		token_emb = self.emb(input_ids)
		position_ids = torch.arange(0, s, device=input_ids.device)
		pos_emb = self.pos(position_ids)
		x = token_emb + pos_emb
		mask = (input_ids != self.mask_id).float() # b, s
		mask = mask.unsqueeze(1).expand(b, s, s)  # b, s, s
		for block in self.blocks:
			x = block(x, mask=mask)
		logits = self.proj(self.ln(x))
		return logits
 
 