import torch 
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeaderAttention(nn.Module):
	def __init__(self,d_model, n_headers):
		super().__init__()
		self.w_q = nn.Linear(d_model, d_model)
		self.w_k = nn.Linear(d_model, d_model)
		self.w_v = nn.Linear(d_model, d_model)
		self.h = d_model // n_headers
		self.n = n_headers
		assert n_headers * self.h == d_model, f"d_model ({d_model}) must be divisible by n_headers ({n_headers})"

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
		assert x.dim() == 3, "input should be a 3D tensor"
		assert x.size(1) == 1, "input should have sequence length of 1 for kv_cache"
		if kv_cache is None:
			x = self.forward(x, mask)
			return x, kv_cache
		else:
			b,s,d = x.size()
			assert len(kv_cache) == 2, "kv_cache should be a tuple of (k, v)"
			for i in range(len(kv_cache)):
				assert kv_cache[i].size(0) == b, "kv_cache should have the same batch size as input"
				assert kv_cache[i].size(1) == n, "kv_cache should have the same number of headers as input"
				assert kv_cache[i].size(2) < s, "kv_cache should have a smaller sequence length than input"
				assert kv_cache[i].size(3) == h, "kv_cache should have the same head size as input"
			k, v = kv_cache # b, n, s, h
			h = self.h
			n = self.n

			# linear projection
			q = self.w_q(x)
			

			# split and transpose
			q = q.view(b,s,n,h).transpose(1,2) # b,n,1,h
			# calculate k and v for new x
			k = torch.cat([k, self.w_k(x).view(b,1,n,h).transpose(1,2)], dim=2) # b,n,s+1,h
			v = torch.cat([v, self.w_v(x).view(b,1,n,h).transpose(1,2)], dim=2) # b,n,s+1,h
			# kv_cache is now updated
			kv_cache = (k, v)

			# matmul, mask, softmax and qkv
			qk = torch.matmul(q, k.transpose(-1,-2)) / math.sqrt(h) # b,n,1,s+1
	
			qk = F.softmax(qk, dim = -1)
			qkv = torch.matmul(qk, v) # b,n,s+1,h

			# transpose, view, and projection
			qkv = qkv.transpose(1,2).contiguous().view(b,s,d)
			return self.w_o(qkv), kv_cache

			
class FeedForward(nn.Module):
	def __init__(self, d_model, d_ff, drop_out_rate):
		super().__init__()
		self.fn1 = nn.Linear(d_model, d_ff)
		self.dropout = nn.Dropout(drop_out_rate)
		self.fn2 = nn.Linear(d_ff, d_model)
		self.ln = nn.LayerNorm(d_model)
	def forward(self, x):
		residue = x
		x = self.fn1(x)
		x = F.relu(x)
		x = self.dropout(x)
		x = self.fn2(x)
		x = self.ln(x + residue)
		return x

class TransformerBlock(nn.Module):
	def __init__(self,d_model, drop_out_rate, d_ff, n_headers):
		super().__init__()
		self.att = MultiHeaderAttention(d_model, n_headers)
		self.ln = nn.LayerNorm(d_model)
		self.ff = FeedForward(d_model,d_ff, drop_out_rate)
	def forward(self, x, mask=None):
		residue = x
		x = self.att(x, mask)
		x = self.ln(x + residue)
		x = self.ff(x)
		return x
	def forward_with_kv_cache(self, x, kv_cache=None):
		residue = x
		x, kv_cache = self.att.forward_with_kv_cache(x, kv_cache)
		x = self.ln(x + residue)
		x = self.ff(x)
		return x, kv_cache

class DecoderOnly(nn.Module):
	def __init__(self, v_size, max_seq, d_model, drop_out_rate, d_ff, n_blocks, n_headers):
		super().__init__()
		self.ms = max_seq
		self.emb = nn.Embedding(v_size, d_model)
		self.pos = nn.Embedding(max_seq, d_model)
		self.blocks = nn.ModuleList([TransformerBlock(d_model, drop_out_rate, d_ff, n_headers) for _ in range(n_blocks)])
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

		# final projection
		logits = self.proj(x)
		return logits

	def forward_with_kv_cache(self, input_ids, kv_cache=None):
		b, s = input_ids.size()
		token_emb = self.emb(input_ids)
		position_ids = torch.arange(0, s, device=input_ids.device).unsqueeze(0)  # 1,
		pos_emb = self.pos(position_ids)
		x = token_emb + pos_emb

		mask = torch.tril(torch.ones(s, s, device=input_ids.device))
		if kv_cache is None:
			x = block(x, mask)
		for i, block in enumerate(self.blocks):
			x, kv_cache[i] = block.forward_with_kv_cache(x, kv_cache[i])

		logits = self.proj(x)
		return logits, kv_cache
 