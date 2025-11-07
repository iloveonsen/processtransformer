
import torch
import torch.nn as nn
import math

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = nn.MultiheadAttention(embed_dim, num_heads, dropout=rate, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_dim)
        )
        self.layernorm_a = nn.LayerNorm(embed_dim, eps=1e-6)
        self.layernorm_b = nn.LayerNorm(embed_dim, eps=1e-6)
        self.dropout_a = nn.Dropout(rate)
        self.dropout_b = nn.Dropout(rate)

    def forward(self, inputs):
        # MultiheadAttention expects (batch, seq, embed_dim) with batch_first=True
        attn_output, _ = self.att(inputs, inputs, inputs, need_weights=False)
        attn_output = self.dropout_a(attn_output)
        out_a = self.layernorm_a(inputs + attn_output)
        ffn_output = self.ffn(out_a)
        ffn_output = self.dropout_b(ffn_output)
        return self.layernorm_b(out_a + ffn_output)

class TokenAndPositionEmbedding(nn.Module):
    def __init__(self, maxlen, vocab_size, embed_dim):
        super(TokenAndPositionEmbedding, self).__init__()
        self.token_emb = nn.Embedding(vocab_size, embed_dim)
        self.pos_emb = nn.Embedding(maxlen, embed_dim)
        self.maxlen = maxlen

    def forward(self, x):
        # x shape: (batch_size, seq_len)
        seq_len = x.size(1)
        positions = torch.arange(0, seq_len, dtype=torch.long, device=x.device)
        positions = self.pos_emb(positions)  # (seq_len, embed_dim)
        x = self.token_emb(x)  # (batch_size, seq_len, embed_dim)
        return x + positions.unsqueeze(0)  # broadcast positions across batch

class NextActivityModel(nn.Module):
    def __init__(self, max_case_length, vocab_size, output_dim,
                 embed_dim=36, num_heads=4, ff_dim=64):
        super(NextActivityModel, self).__init__()
        self.embedding = TokenAndPositionEmbedding(max_case_length, vocab_size, embed_dim)
        self.transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)
        self.dropout1 = nn.Dropout(0.1)
        self.dense1 = nn.Linear(embed_dim, 64)
        self.dropout2 = nn.Dropout(0.1)
        self.output_layer = nn.Linear(64, output_dim)

    def forward(self, inputs):
        x = self.embedding(inputs)
        x = self.transformer_block(x)
        x = torch.mean(x, dim=1)  # Global Average Pooling
        x = self.dropout1(x)
        x = torch.relu(self.dense1(x))
        x = self.dropout2(x)
        outputs = self.output_layer(x)
        return outputs

class NextTimeModel(nn.Module):
    def __init__(self, max_case_length, vocab_size, output_dim=1,
                 embed_dim=36, num_heads=4, ff_dim=64):
        super(NextTimeModel, self).__init__()
        self.embedding = TokenAndPositionEmbedding(max_case_length, vocab_size, embed_dim)
        self.transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)
        self.time_dense = nn.Linear(3, 32)
        self.dropout1 = nn.Dropout(0.1)
        self.dense1 = nn.Linear(embed_dim + 32, 128)
        self.dropout2 = nn.Dropout(0.1)
        self.output_layer = nn.Linear(128, output_dim)

    def forward(self, inputs, time_inputs):
        x = self.embedding(inputs)
        x = self.transformer_block(x)
        x = torch.mean(x, dim=1)  # Global Average Pooling
        x_t = torch.relu(self.time_dense(time_inputs))
        x = torch.cat([x, x_t], dim=1)
        x = self.dropout1(x)
        x = torch.relu(self.dense1(x))
        x = self.dropout2(x)
        outputs = self.output_layer(x)
        return outputs

class RemainingTimeModel(nn.Module):
    def __init__(self, max_case_length, vocab_size, output_dim=1,
                 embed_dim=36, num_heads=4, ff_dim=64):
        super(RemainingTimeModel, self).__init__()
        self.embedding = TokenAndPositionEmbedding(max_case_length, vocab_size, embed_dim)
        self.transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)
        self.time_dense = nn.Linear(3, 32)
        self.dropout1 = nn.Dropout(0.1)
        self.dense1 = nn.Linear(embed_dim + 32, 128)
        self.dropout2 = nn.Dropout(0.1)
        self.output_layer = nn.Linear(128, output_dim)

    def forward(self, inputs, time_inputs):
        x = self.embedding(inputs)
        x = self.transformer_block(x)
        x = torch.mean(x, dim=1)  # Global Average Pooling
        x_t = torch.relu(self.time_dense(time_inputs))
        x = torch.cat([x, x_t], dim=1)
        x = self.dropout1(x)
        x = torch.relu(self.dense1(x))
        x = self.dropout2(x)
        outputs = self.output_layer(x)
        return outputs

def get_next_activity_model(max_case_length, vocab_size, output_dim,
                            embed_dim=36, num_heads=4, ff_dim=64):
    return NextActivityModel(max_case_length, vocab_size, output_dim,
                            embed_dim, num_heads, ff_dim)

def get_next_time_model(max_case_length, vocab_size, output_dim=1,
                       embed_dim=36, num_heads=4, ff_dim=64):
    return NextTimeModel(max_case_length, vocab_size, output_dim,
                        embed_dim, num_heads, ff_dim)

def get_remaining_time_model(max_case_length, vocab_size, output_dim=1,
                             embed_dim=36, num_heads=4, ff_dim=64):
    return RemainingTimeModel(max_case_length, vocab_size, output_dim,
                             embed_dim, num_heads, ff_dim)
