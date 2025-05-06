import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # Linear layers for Q, K, V
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def split_heads(self, x):
        """Split the last dimension into (num_heads, d_k)"""
        batch_size, seq_len, d_model = x.size()
        return x.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

    def combine_heads(self, x):
        """Combine the heads back to original shape"""
        batch_size, _, seq_len, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)

    def forward(self, query, key, value, mask=None):
        # Project inputs to Q, K, V
        q = self.W_q(query)
        k = self.W_k(key)
        v = self.W_v(value)

        # Split into multiple heads
        q = self.split_heads(q)
        k = self.split_heads(k)
        v = self.split_heads(v)

        # Calculate attention scores
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)

        # Apply mask if provided
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)

        # Softmax to get attention weights
        attn_weights = F.softmax(attn_scores, dim=-1)

        # Apply attention weights to values
        output = torch.matmul(attn_weights, v)

        # Combine heads back
        output = self.combine_heads(output)

        # Final linear layer
        output = self.W_o(output)

        return output, attn_weights


class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(PositionWiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return x


class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, mask=None):
        # Self attention sublayer
        attn_output, _ = self.self_attn(x, x, x, mask)
        x = x + self.dropout(attn_output)
        x = self.norm1(x)

        # Feed forward sublayer
        ff_output = self.feed_forward(x)
        x = x + self.dropout(ff_output)
        x = self.norm2(x)

        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.enc_dec_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, enc_output, src_mask=None, tgt_mask=None):
        # Self attention sublayer (with target mask)
        attn_output, _ = self.self_attn(x, x, x, tgt_mask)
        x = x + self.dropout(attn_output)
        x = self.norm1(x)

        # Encoder-decoder attention sublayer
        attn_output, _ = self.enc_dec_attn(x, enc_output, enc_output, src_mask)
        x = x + self.dropout(attn_output)
        x = self.norm2(x)

        # Feed forward sublayer
        ff_output = self.feed_forward(x)
        x = x + self.dropout(ff_output)
        x = self.norm3(x)

        return x


class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, num_layers, d_ff, max_len=5000):
        super(TransformerEncoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        self.layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff)
                                     for _ in range(num_layers)])

    def forward(self, x, mask=None):
        # Embedding layer
        x = self.embedding(x)

        # Positional encoding
        x = self.pos_encoding(x)

        # Pass through all encoder layers
        for layer in self.layers:
            x = layer(x, mask)

        return x


class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, num_layers, d_ff, max_len=5000):
        super(TransformerDecoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        self.layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff)
                                     for _ in range(num_layers)])

    def forward(self, x, enc_output, src_mask=None, tgt_mask=None):
        # Embedding layer
        x = self.embedding(x)

        # Positional encoding
        x = self.pos_encoding(x)

        # Pass through all decoder layers
        for layer in self.layers:
            x = layer(x, enc_output, src_mask, tgt_mask)

        return x


class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, num_heads=8,
                 num_layers=6, d_ff=2048, max_len=5000):
        super(Transformer, self).__init__()
        self.encoder = TransformerEncoder(src_vocab_size, d_model, num_heads,
                                          num_layers, d_ff, max_len)
        self.decoder = TransformerDecoder(tgt_vocab_size, d_model, num_heads,
                                          num_layers, d_ff, max_len)
        self.output_layer = nn.Linear(d_model, tgt_vocab_size)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        # Encode source sequence
        enc_output = self.encoder(src, src_mask)

        # Decode target sequence using encoder output
        dec_output = self.decoder(tgt, enc_output, src_mask, tgt_mask)

        # Final output projection
        output = self.output_layer(dec_output)

        return output


def generate_square_subsequent_mask(sz):
    """Generate a square mask for the sequence where the upper triangle is set to -inf"""
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


def create_padding_mask(seq, pad_idx):
    """Create mask for padded sequences"""
    return (seq != pad_idx).unsqueeze(1).unsqueeze(2)


# Example usage
if __name__ == "__main__":
    # Hyperparameters
    src_vocab_size = 10000
    tgt_vocab_size = 8000
    d_model = 512
    num_heads = 8
    num_layers = 6
    d_ff = 2048
    max_len = 5000
    pad_idx = 0

    # Create model
    transformer = Transformer(src_vocab_size, tgt_vocab_size, d_model,
                              num_heads, num_layers, d_ff, max_len)

    # Example inputs
    src = torch.randint(0, src_vocab_size, (32, 100))  # (batch_size, src_seq_len)
    tgt = torch.randint(0, tgt_vocab_size, (32, 90))  # (batch_size, tgt_seq_len)

    # Create masks
    src_mask = create_padding_mask(src, pad_idx)
    tgt_mask = generate_square_subsequent_mask(tgt.size(1))

    # Forward pass
    output = transformer(src, tgt, src_mask, tgt_mask)
    print("Output shape:", output.shape)  # Should be: torch.Size([32, 90, 8000])