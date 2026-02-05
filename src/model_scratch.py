import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiHeadAttention(nn.Module):
    """
    Implements Scaled Dot-Product Attention with multiple heads.

    The 'Original Contribution' of this project lies in the manual implementation
    of the attention mechanism and the Causal Mask.
    """

    def __init__(self, d_model, num_heads, max_len=1024, dropout=0.1):
        """
        Args:
            d_model (int): The embedding dimension (e.g., 256).
            num_heads (int): Number of parallel attention heads.
            max_len (int): Maximum sequence length for the causal mask.
            dropout (float): Dropout probability to prevent overfitting.
        """
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        # Projections for Q, K, V
        self.c_attn = nn.Linear(d_model, 3 * d_model)
        self.c_proj = nn.Linear(d_model, d_model)

        # Regularization
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        # Causal mask (Lower Triangular Matrix)
        # We register it as a buffer so it's part of state_dict but not a parameter
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(max_len, max_len)).view(1, 1, max_len, max_len),
        )

    def forward(self, x):
        """
        Forward pass for Multi-Head Attention.

        Steps:
        1. Linear projection to get Query (Q), Key (K), Value (V).
        2. Split heads to shape (Batch, Heads, Time, Head_Dim).
        3. Calculate Scaled Dot-Product Attention: softmax(Q @ K^T / sqrt(d_k)).
        4. Apply CAUSAL MASK to ensure position t can only attend to 0...t.
        5. Weighted sum of Values (V).
        6. Reassemble heads and project back to d_model.
        """
        B, T, C = x.size()  # Batch, Time (seq_len), Channels (d_model)

        # Calculate Q, K, V
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.d_model, dim=2)

        # Split heads: (B, T, C) -> (B, T, H, D) -> (B, H, T, D)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        # Attention score: (Q @ K^T) / sqrt(d_k)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))

        # Apply Causal Mask: Mask future tokens (where bias == 0) with -infinity
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))

        # Softmax and weighted sum
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)  # Apply dropout to attention weights

        y = att @ v  # (B, H, T, T) @ (B, H, T, D) -> (B, H, T, D)

        # Reassemble heads: (B, H, T, D) -> (B, T, H, D) -> (B, T, C)
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # Final projection with residual dropout
        return self.resid_dropout(self.c_proj(y))


class FeedForward(nn.Module):
    """
    Standard Position-wise Feed-Forward Network (FFN).
    Project -> Act (GELU) -> Project.
    """

    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    """
    A single Transformer Decoder Block.
    Structure: Input -> LayerNorm -> Attention -> Residual -> LayerNorm -> FFN -> Residual
    """

    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, num_heads, dropout=dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, dropout=dropout)

    def forward(self, x):
        # Residual connections
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class ScratchTransformer(nn.Module):
    """
    The Main 'From-Scratch' Model.
    A Decoder-only Transformer optimized for Next-Word Prediction.

    Components:
    1. Token Embeddings + Positional Embeddings
    2. Stack of TransformerBlocks (Attention + FFN)
    3. Final LayerNorm
    4. Language Modeling Head (Linear layer projecting to vocab_size)
    """

    def __init__(
        self,
        vocab_size,
        d_model=256,
        num_heads=4,
        num_layers=6,
        max_len=1024,
        dropout=0.1,
    ):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_len, d_model)
        self.emb_dropout = nn.Dropout(dropout)

        self.blocks = nn.Sequential(
            *[
                TransformerBlock(d_model, num_heads, dropout=dropout)
                for _ in range(num_layers)
            ]
        )
        self.ln_f = nn.LayerNorm(d_model)  # Final layer norm
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # Weight tying (optional but good practice)
        self.token_embedding.weight = self.lm_head.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        """
        Forward pass for training or inference.

        Args:
            idx (torch.Tensor): Input token indices (Batch, Time).
            targets (torch.Tensor, optional): Target token indices for Loss calculation.

        Returns:
            logits (torch.Tensor): Raw predictions (Batch, Time, Vocab_Size).
            loss (torch.Tensor or None): CrossEntropyLoss if targets provided.
        """
        B, T = idx.size()

        # Create position indices [0, 1, ..., T-1]
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)

        # Embeddings
        tok_emb = self.token_embedding(idx)  # (B, T, C)
        pos_emb = self.position_embedding(pos)  # (T, C)
        x = self.emb_dropout(tok_emb + pos_emb)

        # Transformer Blocks
        x = self.blocks(x)
        x = self.ln_f(x)

        # Logits
        logits = self.lm_head(x)  # (B, T, vocab_size)

        loss = None
        if targets is not None:
            # Flatten for CrossEntropyLoss
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss


if __name__ == "__main__":
    print("Running Causal Mask Unit Test...")

    vocab_size = 100
    model = ScratchTransformer(vocab_size=vocab_size, d_model=32, num_heads=2, num_layers=1)
    model.eval()

    # Input sequence: [10, 20, 30]
    input_seq = torch.tensor([[10, 20, 30]])

    # Get output for the FIRST token (position 0)
    with torch.no_grad():
        output_1, _ = model(input_seq)
        first_token_out_1 = output_1[0, 0, :].clone()

    # Change the LAST token: [10, 20, 99]
    # The output for position 0 (which sees only '10') should NOT change.
    input_seq_modified = torch.tensor([[10, 20, 99]])

    with torch.no_grad():
        output_2, _ = model(input_seq_modified)
        first_token_out_2 = output_2[0, 0, :].clone()

    # Check if they are identical
    diff = torch.sum(torch.abs(first_token_out_1 - first_token_out_2))
    print(f"Difference in 1st token output after changing 3rd token: {diff.item()}")

    if diff.item() < 1e-6:
        print("PASSED: Causal Mask is working correctly.")
    else:
        print("FAILED: Future tokens are leaking information!")
