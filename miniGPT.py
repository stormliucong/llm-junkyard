import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiAttention(nn.Module):
	def __init__(self, n_header, d_model):
		super().__init__()
		self.d_model = d_model
		self.n_header = n_header

		self.w_q = nn.Linear(d_model, d_model)
		self.w_k = nn.Linear(d_model, d_model)
		self.w_v = nn.Linear(d_model, d_model)
		self.w_o = nn.Linear(d_model, d_model)

		self.h_model = self.d_model // self.n_header
		assert self.h_model * self.n_header == self.d_model


	def forward(self, x, mask=None):
		batch_size, seq_length, d_model = x.size()
		n_header = self.n_header
		h_model = self.h_model

		# linear projection
		Q = self.w_q(x)
		K = self.w_k(x)
		V = self.w_v(x)

		# split and transpose
		Q = Q.view(batch_size, seq_length, n_header, h_model).transpose(1,2).contiguous()
		K = K.view(batch_size, seq_length, n_header, h_model).transpose(1,2).contiguous()
		V = V.view(batch_size, seq_length, n_header, h_model).transpose(1,2).contiguous() # b,n,s,h

		# QK
		QK = torch.matmul(Q,K.transpose(-1,-2)) / math.sqrt(h_model) # b,n,s,s

		# mask 
		if mask is not None:
			QK = QK.masked_fill(mask == 0, -1e-9)

		# softmax
		QK = torch.softmax(QK, dim=-1)

		# QKV
		QKV = torch.matmul(QK, V).transpose(1,2).contiguous().view(batch_size, seq_length, d_model) # b,s,d

		# output
		return self.w_o(QKV)

class FeedForward(nn.Module):
	def __init__(self, d_model, d_ff):
		super().__init__()
		self.ff1 = nn.Linear(d_model, d_ff)
		self.ff2 = nn.Linear(d_ff, d_model)

	def forward(self, x):
		return self.ff2(F.ReLU(self.ff1(x)))


class TransformerBlock(nn.Module):
	def __init__(self, n_header, d_model, d_ff, dropout_rate):
		super().__init__()
		self.attention = MultiAttention(n_header, d_model)
		self.ff = FeedForward(d_model, d_ff)
		self.ln1 = nn.LayerNorm(d_model)
		self.ln2 = nn.LayerNorm(d_model)
		self.dropout = nn.Dropout(dropout_rate)


	def forward(self, x, mask=None):
		o1 = self.attention(x, mask)
		o1 = self.ln1(x + self.dropout(o1))
		o2 = self.ff(o1)
		o2 = self.ln2(o1 + self.dropout(o2))
		
		return o2

class Decoder(nn.Module):

	def __init__(self,n_header, d_model, d_ff, dropout_rate, vocabulary_size, max_sequence_length, n_layer):
		super().__init__()
		self.emb = nn.Embedding(vocabulary_size, d_model)
		self.position = nn.Embedding(max_sequence_length, d_model)
		self.blocks = nn.ModuleList([TransformerBlock(n_header, d_model, d_ff, dropout_rate) for _ in range(self.n_layer)])
		self.project = nn.Linear(d_model, vocabulary_size, bias=False)

		self.mask = torch.tril(torch.ones())


	def forward(self, input_ids):

		batch_size, seq_length, d_model = input_ids.size()

		token_emb = self.emb(input_ids) # b,s,d

		position_emb = self.emb(self.max_sequence_length, d_model).unsqueeze(0).unsqueeze(0) # 1,1,d

		x = token_emb + position_emb

		for block in self.blocks:
			x = block(x, self.mask)

		# final projection
		logits = nn.project(x) # b,s,v

	return logits