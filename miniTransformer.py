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

	def forward(self, q, k, v, mask=None):
		b,s_q, d = q.size()
		b,s_k, _ = k.size()
		b,s_v, _ = v.size()
		h = self.h
		n = self.n
  
		if mask is not None:
			assert mask.dim() == 3, "mask should be a 3D tensor"
			assert mask.size(0) == b, "mask should have the same batch size as input q"
        
        
		# linear projection
		q = self.w_q(q)
		k = self.w_k(k)
		v = self.w_v(v)

		# split and transpose
		q = q.view(b,s_q,n,h).transpose(1,2) # b,n,s_q,h
		k = k.view(b,s_k,n,h).transpose(1,2) # b,n,s_k,h
		v = v.view(b,s_v,n,h).transpose(1,2) # b,n,s_v,h

		# matmul, mask, softmax and qkv
		qk = torch.matmul(q, k.transpose(-1,-2)) / math.sqrt(h) # b,n,s_q,s_k

		if mask is not None:
			mask = mask.unsqueeze(1)  # b, 1, s_q, s_k
			qk.masked_fill_(mask == 0, -1e9) 
		qk = F.softmax(qk, dim = -1)
		qkv = torch.matmul(qk, v) # b,n,s_q,h

		# transpose, view, and projection
		qkv = qkv.transpose(1,2).contiguous().view(b,s_q,d)

		return self.w_o(qkv)
			
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

class EncoderBlock(nn.Module):
    def __init__(self, d_model, drop_out_rate, d_ff, n_heads):
        super().__init__()
        self.att = MultiHeaderAttention(d_model, n_heads)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, d_ff, drop_out_rate)
    def forward(self, x, mask=None):
        residue1 = x
        x = self.ln1(x)
        x = self.att(x, x, x, mask)
        x = x + residue1
        residue2 = x
        x = self.ln2(x)
        x = self.ff(x)
        x = x + residue2
        return x

class DecoderBlock(nn.Module):
	def __init__(self,d_model, drop_out_rate, d_ff, n_heads):
		super().__init__()
		self.cross_att = MultiHeaderAttention(d_model, n_heads)
		self.self_att = MultiHeaderAttention(d_model, n_heads)
		self.ln1 = nn.LayerNorm(d_model)
		self.ln2 = nn.LayerNorm(d_model)
		self.ln3 = nn.LayerNorm(d_model)
		self.ff = FeedForward(d_model, d_ff, drop_out_rate)

	def forward(self, x, encoder_output, self_attention_mask=None, cross_attention_mask=None):
		residue1 = x # pre-layer norm
		x = self.ln1(x)
		x = self.self_att(x, x, x, self_attention_mask) # self-attention
		x = self.ln1(residue1 + x)
		residue2 = x 
		
		if encoder_output is not None:
			x = self.cross_att(x, encoder_output, encoder_output, cross_attention_mask)  # cross-attention
			x = self.ln2(x + residue2)
			x = x + residue2
			residue3 = x   
			x = self.ln3(self.ff(x))
			x = x + residue3
			return x
		else:
			x = self.ln2(x)
			x = self.ff(x)
			return x + residue1

class Encoder(nn.Module):
    def __init__(self, v_size, max_seq, d_model, drop_out_rate, d_ff, n_blocks, n_heads, mask_id=None):
        super().__init__()
        self.ms = max_seq
        self.emb = nn.Embedding(v_size, d_model)
        self.pos = nn.Embedding(max_seq, d_model)
        self.blocks = nn.ModuleList([EncoderBlock(d_model, drop_out_rate, d_ff, n_heads) for _ in range(n_blocks)])
        self.ln = nn.LayerNorm(d_model)
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
        
        # final layernorm
        x = self.ln(x)
        return x

class Decoder(nn.Module):
    def __init__(self, v_size, max_seq, d_model, drop_out_rate, d_ff, n_blocks, n_heads, mask_id=None):
        super().__init__()
        self.ms = max_seq
        self.emb = nn.Embedding(v_size, d_model)
        self.pos = nn.Embedding(max_seq, d_model)
        self.blocks = nn.ModuleList([DecoderBlock(d_model, drop_out_rate, d_ff, n_heads) for _ in range(n_blocks)])
    
    def forward(self, output_ids, encoder_output=None):
        b, s = output_ids.size()
        assert s <= self.ms, f"Input sequence length {s} exceeds maximum sequence length {self.ms}"
        token_emb = self.emb(output_ids)
        position_ids = torch.arange(0, s, device=output_ids.device)
        pos_emb = self.pos(position_ids)
        x = token_emb + pos_emb
        causal_mask = torch.tril(torch.ones(s,s, device=output_ids.device)) # s, s
        mask = (output_ids != self.mask_id).float() # b, s
        mask = mask.unsqueeze(1).expand(b, s, s)  # b, s, s
        mask = mask & causal_mask.unsqueeze(0)  # b, s, s
        for block in self.blocks:
            x = block(x, encoder_output=encoder_output, mask=mask)
        # final layernorm and projection
        x = self.ln(x)
        x = self.proj(x)
        return x
    
class Transformer(nn.Module):
    def __init__(self, v_size, max_seq, d_model, drop_out_rate, d_ff, n_blocks, n_heads, mask_id=None):
        super().__init__()
        self.ms = max_seq
        self.encoder = Encoder(v_size, max_seq, d_model, drop_out_rate, d_ff, n_blocks, n_heads, mask_id=mask_id)
        self.decoder = Decoder(v_size, max_seq, d_model, drop_out_rate, d_ff, n_blocks, n_heads, mask_id=mask_id)

    def forward(self, input_ids, output_ids):
        encoder_output = self.encoder(input_ids)
        # Pass the encoder output to the decoder
        decoder_output = self.decoder(output_ids, encoder_output=encoder_output)
        return decoder_output
