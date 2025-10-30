import torch 
import torch.nn as nn
import math
from typing import Optional

#masking
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model:int, n_heads:int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be devisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        
        #projections for Q K V
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)
        
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T, C = x.shape #batch, sequence len, d_model
        
        #projections
        Q = self.q_proj(x)  # (B, T, d_model)
        K = self.k_proj(x)
        V = self.v_proj(x)
        
        Q = Q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        
        #SDPA
        attn_scores = (Q @ K.transpose(-2, -1)) / self.scale #(B, n_heads, T, T)
        
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
        
        attn_weights = torch.softmax(attn_scores, dim = -1)
        attn_weights = self.dropout(attn_weights)
        
        #weighted summ of values
        output = attn_weights @ V
        output = output.transpose(1, 2).contiguous().view(B, T, self.d_model)
        
        return self.out_proj(output)


#feed foorward network
class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int=None, dropout: float = 0.1):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )    
        
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        return self.net(x)
    
#one block of Decoder with no cross-attention
class DecoderBlock(nn.Module):
    def __init__(self, d_model: int, n_heads:int, d_ff:int = None, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        #self-attention and residual
        attn_out = self.self_attn(x, mask)
        x = x+ self.dropout(attn_out)
        x = self.ln1(x)
        
        #feed-forward + residual
        ff_out = self.ff(x)
        x = x + self.dropout(ff_out)
        x = self.ln2(x)
        
        return x

class TransformerDecoderLM(nn.Module):
    def __init__(self, vocab_size:int, d_model: int = 256, n_layers:int=4, n_heads: int = 8, d_ff: int = None, max_seq_len: int = 512, dropout: float = 0.1, tie_weights:bool = True):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        
        #embs
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model) #learnable PE
        
        #decoder layers
        self.layers = nn.ModuleList([
            DecoderBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

        self.ln_final = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
        #output layer
        self.lm_head = nn.Linear(d_model, vocab_size, bias = False)
        
        if tie_weights:
            self.lm_head.weight = self.token_emb.weight
            
        self._init_weights()
    
    def _init_weights(self):
        #initialization like in GPT-2
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        
    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        B, T = tokens.shape
        assert T <= self.max_seq_len, f"Sequence too long: {T} > {self.max_seq_len}"
        
        #embs
        x = self.token_emb(tokens) * math.sqrt(self.d_model)#scalling like in originl
        pos = self.pos_emb(torch.arange(T, device=tokens.device))
        x = x + pos
        x = self.dropout(x)
        
        #masking(casual)
        mask = torch.tril(torch.ones(T, T, device=tokens.device)).unsqueeze(0).unsqueeze(0)  # (1, 1, T, T)
        
        #going on layers
        for layer in self.layers:
            x = layer(x, mask)
            
        x = self.ln_final(x)
        logits = self.lm_head(x)# (B, T, vocab_size)
        return logits