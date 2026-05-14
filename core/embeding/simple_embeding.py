import numpy as np
from typing import List, Optional

class RoPE:
    """Rotary Positional Embedding"""
    def __init__(self, dim: int, max_seq_len: int = 2048, theta: float = 10000.0):
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.theta = theta
        
        # Precompute frequencies
        inv_freq = 1.0 / (theta ** (np.arange(0, dim, 2).astype(np.float32) / dim))
        self.inv_freq = inv_freq

    def forward(self, x: np.ndarray, offset: int = 0) -> np.ndarray:
        """
        x shape: (batch_size, seq_len, dim) atau (seq_len, dim)
        """
        seq_len = x.shape[-2]
        pos = np.arange(offset, offset + seq_len, dtype=np.float32)
        
        # Compute angles
        freqs = np.outer(pos, self.inv_freq)           # (seq_len, dim//2)
        emb = np.concatenate((freqs, freqs), axis=-1)  # (seq_len, dim)
        
        cos = np.cos(emb)
        sin = np.sin(emb)
        
        # Reshape untuk broadcasting
        if x.ndim == 2:  # (seq_len, dim)
            cos = cos.reshape(seq_len, 1, -1) if x.ndim == 2 else cos
            sin = sin.reshape(seq_len, 1, -1)
            x_rot = self._rotate_half(x)
            return x * cos + x_rot * sin
        else:  # (batch, seq_len, dim)
            cos = cos.reshape(1, seq_len, 1, -1)
            sin = sin.reshape(1, seq_len, 1, -1)
            x_rot = self._rotate_half(x)
            return x * cos + x_rot * sin

    def _rotate_half(self, x: np.ndarray) -> np.ndarray:
        """Rotate half the hidden dims"""
        x1 = x[..., :x.shape[-1]//2]
        x2 = x[..., x.shape[-1]//2:]
        return np.concatenate((-x2, x1), axis=-1)


class Embedding:
    """
    Token Embedding + RoPE
    """
    def __init__(self, vocab_size: int, embed_dim: int = 512, 
                 max_seq_len: int = 2048, padding_idx: Optional[int] = None):
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len
        self.padding_idx = padding_idx

        # Token Embedding
        np.random.seed(42)
        scale = 0.02
        self.token_embedding = np.random.randn(vocab_size, embed_dim).astype(np.float32) * scale
        
        # RoPE
        self.rope = RoPE(dim=embed_dim, max_seq_len=max_seq_len)

        print(f"✅ Embedding Layer dibuat:")
        print(f"   Vocab Size     : {vocab_size:,}")
        print(f"   Embedding Dim  : {embed_dim}")
        print(f"   Max Seq Length : {max_seq_len}")
        print(f"   RoPE enabled   : Yes\n")

    def forward(self, token_ids: List[int]) -> np.ndarray:
        """Single sequence"""
        token_ids = np.array(token_ids, dtype=np.int32)
        
        # Token Embedding
        x = self.token_embedding[token_ids]          # (seq_len, embed_dim)
        
        # Apply RoPE
        x = self.rope.forward(x)
        
        return x

    def forward_batch(self, batch_ids: List[List[int]]) -> np.ndarray:
        """Multiple sequences (batch)"""
        batch_size = len(batch_ids)
        seq_len = max(len(ids) for ids in batch_ids)
        
        batch_emb = np.zeros((batch_size, seq_len, self.embed_dim), dtype=np.float32)
        
        for i, ids in enumerate(batch_ids):
            emb = self.forward(ids)
            batch_emb[i, :len(emb)] = emb
            
        return batch_emb
    
class RMSNorm:
    def __init__(self, dim: int, eps: float = 1e-6):
        self.eps = eps
        self.weight = np.ones(dim, dtype=np.float32)  # gamma (learnable)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        x shape: (batch_size, seq_len, dim) atau (seq_len, dim)
        """
        # Hitung RMS
        rms = np.sqrt(np.mean(x ** 2, axis=-1, keepdims=True) + self.eps)
        
        # Normalisasi + scaling
        x_norm = x / rms
        
        return x_norm * self.weight