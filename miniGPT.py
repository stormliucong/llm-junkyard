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
		

	def forward(self, x, mask=None):
		b,s,d = x.size()
		h = self.h
		n = self.n

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
	def forward(self, x):
		residue = x
		x = self.fn1(x)
		x = F.relu(x)
		x = self.dropout(x)
		x = self.fn2(x)
		# x = self.ln(x + residue)
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

class DecoderOnly(nn.Module):
	def __init__(self, v_size, max_seq, d_model, drop_out_rate, d_ff, n_blocks, n_heads):
		super().__init__()
		self.ms = max_seq
		self.emb = nn.Embedding(v_size, d_model)
		self.pos = nn.Embedding(max_seq, d_model)
		self.blocks = nn.ModuleList([TransformerBlock(d_model, drop_out_rate, d_ff, n_heads) for _ in range(n_blocks)])
		self.ln = nn.LayerNorm(d_model)
		# final projection
  		self.proj = nn.Linear(d_model, v_size)

	def forward(self, input_ids):
		b, s = input_ids.size()
		token_emb = self.emb(input_ids)
		position_ids = torch.arange(0, s, device = input_ids.device).unsqueeze(0) # 1,
		pos_emb = self.pos(position_ids)
		x = token_emb + pos_emb
		mask = torch.tril(torch.ones(s,s, device=input_ids.device))
		for block in self.blocks:
			x = block(x, mask)

		# final layernorm and projection
		logits = self.ln(self.proj(x))
		return logits

	def forward_with_kv_cache(self, input_ids, kv_cache=None):
		
		b, s = input_ids.size()

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
		logits = self.ln(self.proj(x))
		return logits, kv_cache
 