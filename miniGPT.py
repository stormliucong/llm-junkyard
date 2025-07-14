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
		assert n_headers * self.h == d_model

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
		x = self.drop(x)
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
		
		
