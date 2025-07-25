import torch 
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeaderAttention(nn.Module):
    def __init__(self, d_model, n_heads):
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

    def forward(self, q, k, v, mask=None, return_attention=False):
        b, s_q, d = q.size()
        b, s_k, _ = k.size()
        b, s_v, _ = v.size()
        h = self.h
        n = self.n     

        # linear projection
        q = self.w_q(q)
        k = self.w_k(k)
        v = self.w_v(v)

        # split and transpose
        q = q.view(b, s_q, n, h).transpose(1, 2)  # b,n,s_q,h
        k = k.view(b, s_k, n, h).transpose(1, 2)  # b,n,s_k,h
        v = v.view(b, s_v, n, h).transpose(1, 2)  # b,n,s_v,h

        # matmul, mask, softmax and qkv
        qk = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(h)  # b,n,s_q,s_k

        if mask is not None:
            # mask should be of shape (b, 1, s_q, s_k)
            if mask.dim() == 2:  # if mask is of shape (b, s_k)
                mask = mask.unsqueeze(1).unsqueeze(2)  # b, 1, 1, s_k
            elif mask.dim() == 3:  # if mask is of shape (b, s_q, s_k)
                mask = mask.unsqueeze(1)  # b, 1, s_q, s_k
            elif mask.dim() == 4:  # if mask is already of shape (b, 1, s_q, s_k)
                pass  # mask is already in the correct shape
            else:
                raise ValueError("Mask must be of shape (b, s_k) or (b, s_q, s_k) or (b, 1, s_q, s_k)")
            # ensure mask is broadcastable to qk
            qk.masked_fill_(mask == 0, -1e9)
        qk = F.softmax(qk, dim=-1)
        qkv = torch.matmul(qk, v)  # b,n,s_q,h

        # transpose, view, and projection
        qkv = qkv.transpose(1, 2).contiguous().view(b, s_q, d)
        if return_attention:
            return self.w_o(qkv), qk
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
    def forward(self, x, mask=None, return_attention=False):
        residue1 = x
        x = self.ln1(x)
        if return_attention:
            x, encoder_attention = self.att(x, x, x, mask)
        else:
            x = self.att(x, x, x, mask)     
        x = x + residue1
        residue2 = x
        x = self.ln2(x)
        x = self.ff(x)
        x = x + residue2
        if return_attention:
            return x, encoder_attention
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

    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None, return_attention=False):
        residue1 = x # pre-layer norm
        x = self.ln1(x)
        decoder_self_attention = None
        decoder_cross_attention = None
        if return_attention:
            x, decoder_self_attention = self.self_att(x, x, x, tgt_mask, return_attention) # self-attention
        else:
            x = self.self_att(x, x, x, tgt_mask) # self-attention
        x = self.ln1(residue1 + x)
        residue2 = x         
        
        if encoder_output is not None:
            if return_attention:
                x, decoder_cross_attention = self.cross_att(x, encoder_output, encoder_output, src_mask, return_attention)
            else:
                x = self.cross_att(x, encoder_output, encoder_output, src_mask)  # cross-attention
            x = self.ln2(x + residue2)
            x = x + residue2
            residue3 = x   
            x = self.ln3(self.ff(x))
            x = x + residue3
            
        else:
            x = self.ln2(x)
            x = self.ff(x)
            x = x + residue1
        
        if return_attention:
            return x, decoder_self_attention, decoder_cross_attention
        else:
            return x

class Encoder(nn.Module):
    def __init__(self, v_size, max_seq, d_model, drop_out_rate, d_ff, n_blocks, n_heads, mask_id=None):
        super().__init__()
        self.ms = max_seq
        self.emb = nn.Embedding(v_size, d_model)
        self.pos = nn.Embedding(max_seq, d_model)
        self.blocks = nn.ModuleList([EncoderBlock(d_model, drop_out_rate, d_ff, n_heads) for _ in range(n_blocks)])
        self.ln = nn.LayerNorm(d_model)
        self.mask_id = mask_id
    def forward(self, input_ids, src_mask=None, return_attention=False):
        b, s = input_ids.size()
        assert s <= self.ms, f"Input sequence length {s} exceeds maximum sequence length {self.ms}"
        token_emb = self.emb(input_ids)
        position_ids = torch.arange(0, s, device=input_ids.device)
        pos_emb = self.pos(position_ids)
        x = token_emb + pos_emb
        encoder_attentions = []
        for block in self.blocks:
            if return_attention:
                x, encoder_attention = block(x, src_mask, return_attention)
                encoder_attentions.append(encoder_attention)
            else:
                x = block(x, src_mask)

        # final layernorm
        x = self.ln(x)
        if return_attention:
            return x, encoder_attentions
        return x

class Decoder(nn.Module):
    def __init__(self, v_size, max_seq, d_model, drop_out_rate, d_ff, n_blocks, n_heads):
        super().__init__()
        self.ms = max_seq
        self.emb = nn.Embedding(v_size, d_model)
        self.pos = nn.Embedding(max_seq, d_model)
        self.blocks = nn.ModuleList([DecoderBlock(d_model, drop_out_rate, d_ff, n_heads) for _ in range(n_blocks)])
        self.ln = nn.LayerNorm(d_model)
        self.proj = nn.Linear(d_model, v_size)
    
    def forward(self, output_ids, encoder_output=None, src_mask=None, tgt_mask=None, return_attention=False):
        b, s = output_ids.size()
        assert s <= self.ms, f"Input sequence length {s} exceeds maximum sequence length {self.ms}"
        token_emb = self.emb(output_ids)
        position_ids = torch.arange(0, s, device=output_ids.device)
        pos_emb = self.pos(position_ids)
        x = token_emb + pos_emb
        decoder_self_attentions = []
        decoder_cross_attentions = []    
        for block in self.blocks:
            if return_attention:
                x, decoder_self_attention, decoder_cross_attention = block(x, encoder_output, src_mask, tgt_mask, return_attention)
                decoder_self_attentions.append(decoder_self_attention)
                decoder_cross_attentions.append(decoder_cross_attention)
            else:
                x = block(x, encoder_output, src_mask, tgt_mask)
        # final layernorm and projection
        x = self.ln(x)
        x = self.proj(x)
        if return_attention:
            return x, decoder_self_attentions, decoder_cross_attentions
        return x
    
class Transformer(nn.Module):
    def __init__(self, v_size, max_seq, d_model, drop_out_rate, d_ff, n_blocks, n_heads, pad_idx=0):
        super().__init__()
        self.ms = max_seq
        self.pad_idx = pad_idx
        self.encoder = Encoder(v_size, max_seq, d_model, drop_out_rate, d_ff, n_blocks, n_heads)
        self.decoder = Decoder(v_size, max_seq, d_model, drop_out_rate, d_ff, n_blocks, n_heads)
    def _create_padding_mask(self, x):
        return (x != self.pad_idx).unsqueeze(1).unsqueeze(2).to(dtype=torch.bool) # dimensions: (batch_size, 1, 1, seq_length)
    
    def _create_causal_mask(self, size):
        return torch.tril(torch.ones(size, size)).unsqueeze(0).unsqueeze(0).to(dtype=torch.bool) # dimensions: (1, 1, size, size)
    
    def forward(self, input_ids, output_ids=None, return_attention=False):
        src_mask = self._create_padding_mask(input_ids) # b, 1, 1, s
        tgt_mask = self._create_causal_mask(output_ids.size(1)) # b, 1, s+1, s+1
        tgt_mask = tgt_mask.to(output_ids.device)  # Ensure tgt_mask is on the same device as output_ids
        tgt_mask = tgt_mask & self._create_padding_mask(output_ids)  # Combine padding mask with causal mask
        
        if return_attention:
            encoder_output, encoder_attentions = self.encoder(input_ids, src_mask,return_attention)
            # Pass the encoder output to the decoder
            decoder_output, decoder_self_attentions, decoder_cross_attentions = self.decoder(output_ids, encoder_output, src_mask, tgt_mask,return_attention)
            return decoder_output, encoder_attentions, decoder_self_attentions, decoder_cross_attentions
            
        # Pass the input_ids through the encoder
        encoder_output = self.encoder(input_ids, src_mask,return_attention)
        # Pass the encoder output to the decoder
        decoder_output = self.decoder(output_ids, encoder_output, src_mask, tgt_mask,return_attention)
        return decoder_output
    
    def generate(self, input_ids, max_length, start_token_id=None, end_token_id=None):
        self.eval()
        src_mask = self._create_padding_mask(input_ids) # b, 1, 1, s
        encoder_output = self.encoder(input_ids, src_mask)
        
        # initialize output_ids with start_token_id
        output_ids = torch.full((input_ids.size(0), 1), fill_value=start_token_id, dtype=torch.long, device=input_ids.device)
        
        # generate tokens one by one
        for _ in range(max_length):
            tgt_mask = self._create_causal_mask(output_ids.size(1)) # b, 1, s+1, s+1
            tgt_mask = tgt_mask.to(output_ids.device)
            print(f"output_ids: {output_ids}")
            decoder_output = self.decoder(output_ids, encoder_output=encoder_output, src_mask=src_mask, tgt_mask=tgt_mask)
            
            # get the last token's logits
            next_token_logits = decoder_output[:, -1, :]
            next_token_id = next_token_logits.argmax(dim=-1, keepdim=True)
            output_ids = torch.cat([output_ids, next_token_id], dim=1)
            
            # stop if end_token_id is generated
            if (next_token_id == end_token_id).all():
                break
            
        return output_ids    
