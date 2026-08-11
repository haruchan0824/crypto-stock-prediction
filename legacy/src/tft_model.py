#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""
Implementation of Temporal Fusion Transformers: https://arxiv.org/abs/1912.09363
"""


from torch import nn
import math
import torch
import ipdb
import torch.nn.functional as F
import numpy as np
import torch.nn as nn
import pytorch_lightning as pl

class QuantileLoss(nn.Module):
    ## From: https://medium.com/the-artificial-impostor/quantile-regression-part-2-6fdbc26b2629

    def __init__(self, quantiles):
        ##takes a list of quantiles
        super().__init__()
        self.quantiles = quantiles

    def forward(self, preds, target):
        assert not target.requires_grad
        assert preds.size(0) == target.size(0)
        losses = []
        for i, q in enumerate(self.quantiles):
            errors = target - preds[:, i]
            losses.append(
                torch.max(
                   (q-1) * errors,
                   q * errors
            ).unsqueeze(1))
        loss = torch.mean(
            torch.sum(torch.cat(losses, dim=1), dim=1))
        return loss

class TimeDistributed(nn.Module):
    ## Takes any module and stacks the time dimension with the batch dimenison of inputs before apply the module
    ## From: https://discuss.pytorch.org/t/any-pytorch-function-can-work-as-keras-timedistributed/1346/4
    def __init__(self, module, batch_first=False):
        super(TimeDistributed, self).__init__()
        self.module = module
        self.batch_first = batch_first

    def forward(self, x):

        if len(x.size()) <= 2:
            return self.module(x)

        # Squash samples and timesteps into a single axis
        x_reshape = x.contiguous().view(-1, x.size(-1))  # (samples * timesteps, input_size)

        y = self.module(x_reshape)

        # We have to reshape Y
        if self.batch_first:
            y = y.contiguous().view(x.size(0), -1, y.size(-1))  # (samples, timesteps, output_size)
        else:
            y = y.view(-1, x.size(1), y.size(-1))  # (timesteps, samples, output_size)

        return y

class GLU(nn.Module):
    #Gated Linear Unit
    def __init__(self, input_size):
        super(GLU, self).__init__()

        self.fc1 = nn.Linear(input_size,input_size)
        self.fc2 = nn.Linear(input_size, input_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        sig = self.sigmoid(self.fc1(x))
        x = self.fc2(x)
        return torch.mul(sig, x)


class GatedResidualNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout=0.0):
        super(GatedResidualNetwork, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout = dropout

        # Only create layers if input_size is positive
        if input_size > 0:
            self.fc1 = nn.Linear(input_size, hidden_size)
            self.gelu = nn.GELU() # Using GELU as in the paper
            self.dropout_layer = nn.Dropout(dropout)
            self.fc2 = nn.Linear(hidden_size, output_size)
            self.layer_norm = nn.LayerNorm(output_size)

            # For skip connection if input_size != output_size
            if input_size != output_size:
                self.skip_connection = nn.Linear(input_size, output_size)
            else:
                self.skip_connection = nn.Identity()

            # Gating mechanism
            self.gate = nn.Linear(input_size, output_size * 2) # For gates
        else:
            # Define dummy layers or None if input_size is 0
            self.fc1 = None
            self.gelu = None
            self.dropout_layer = None
            self.fc2 = None
            self.layer_norm = None
            self.skip_connection = None
            self.gate = None


    def forward(self, x, additional_input=None):
        # Combine input with additional_input if provided
        if additional_input is not None:
            x_in = torch.cat([x, additional_input], dim=-1)
        else:
            x_in = x

        # If input size was 0, return zero tensor of output size
        if self.input_size == 0 or x_in.size(-1) == 0:
             # Return a tensor with shape (batch_size, seq_len, output_size) or (batch_size, output_size)
             # depending on the input tensor's dimensions.
             # Assuming x_in has at least batch and feature dimensions.
             output_shape = list(x_in.shape[:-1]) + [self.output_size]
             return torch.zeros(*output_shape, device=x_in.device)


        # Apply gating
        gates = self.gate(x_in)
        gate = torch.sigmoid(gates[..., :self.output_size])
        gate_prime = torch.sigmoid(gates[..., self.output_size:])

        # Main network path
        h = self.fc1(x_in)
        h = self.gelu(h)
        h = self.dropout_layer(h)
        h = self.fc2(h)

        # Apply gating to the main network path
        gated_h = h * gate

        # Skip connection
        skip = self.skip_connection(x_in)

        # Combine and normalize
        output = self.layer_norm(gated_h + skip)

        # Apply gate_prime (second gating)
        output = output * gate_prime

        return output

class InterpretableMultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0):
        super(InterpretableMultiHeadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        # Only create layers if embed_dim is positive
        if embed_dim > 0:
            self.wq = nn.Linear(embed_dim, embed_dim, bias=False)
            self.wk = nn.Linear(embed_dim, embed_dim, bias=False)
            self.wv = nn.Linear(embed_dim, embed_dim, bias=False)

            self.out_proj = nn.Linear(embed_dim, embed_dim)
            self.dropout_layer = nn.Dropout(dropout)
            self.layer_norm = nn.LayerNorm(embed_dim)

            # Attention weights interpretation layer (linear combination across heads)
            self.attn_interpret = nn.Linear(num_heads, 1)
        else:
            # Define dummy layers or None if embed_dim is 0
            self.wq = None
            self.wk = None
            self.wv = None
            self.out_proj = None
            self.dropout_layer = None
            self.layer_norm = None
            self.attn_interpret = None


    def forward(self, q, k, v, attn_mask=None):
        # q, k, v shape: (batch_size, seq_len, embed_dim)

        # If embed_dim was 0, return zero tensor of appropriate shape
        if self.embed_dim == 0 or q.size(-1) == 0:
             batch_size, seq_len = q.size(0), q.size(1) if q.ndim > 1 else 1
             # Return output and dummy attention weights
             return torch.zeros(batch_size, seq_len, self.embed_dim, device=q.device), torch.zeros(batch_size, seq_len, seq_len, device=q.device)


        # Linear transformations and split into heads
        # Shape: (batch_size, num_heads, seq_len, head_dim)
        q = self.wq(q).view(q.size(0), q.size(1), self.num_heads, self.head_dim).transpose(1, 2)
        k = self.wk(k).view(k.size(0), k.size(1), self.num_heads, self.head_dim).transpose(1, 2)
        v = self.wv(v).view(v.size(0), v.size(1), self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled Dot-Product Attention
        # (batch_size, num_heads, seq_len, head_dim) @ (batch_size, num_heads, head_dim, seq_len)
        # -> (batch_size, num_heads, seq_len, seq_len)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)

        if attn_mask is not None:
            attn_scores = attn_scores.masked_fill(attn_mask == 0, float('-inf'))

        # Apply softmax to get attention weights
        attn_weights = F.softmax(attn_scores, dim=-1) # Shape: (batch_size, num_heads, seq_len, seq_len)
        # attn_weights = self.dropout_layer(attn_weights) # Dropout on attention weights (optional, but common)

        # Multiply attention weights with values
        # (batch_size, num_heads, seq_len, seq_len) @ (batch_size, num_heads, seq_len, head_dim)
        # -> (batch_size, num_heads, seq_len, head_dim)
        attn_output = torch.matmul(attn_weights, v)

        # Concatenate heads and apply final linear layer
        # Shape: (batch_size, seq_len, num_heads * head_dim = embed_dim)
        attn_output = attn_output.transpose(1, 2).contiguous().view(attn_output.size(0), -1, self.embed_dim)

        # Residual connection and normalization
        # Ensure the residual connection shape matches after the linear layer
        # If the output of the linear layer has the shape (batch_size, seq_len, embed_dim),
        # the skip connection should also be (batch_size, seq_len, embed_dim).
        # The input to the layer_norm is attn_output from the linear layer PLUS the residual.
        # The residual connection here is from the input 'q' (before linear transform and split)
        # Reshape 'q' back to (batch_size, seq_len, embed_dim) for residual
        residual_q = q.transpose(1, 2).contiguous().view(q.size(0), -1, self.embed_dim)
        output = self.layer_norm(attn_output + self.dropout_layer(self.out_proj(attn_output))) # Original paper seems to add residual before layer_norm

        # For interpretability: Sum attention weights across heads (after softmax)
        # Shape: (batch_size, num_heads, seq_len_q, seq_len_k) -> (batch_size, seq_len_q, seq_len_k, num_heads)
        # Then apply linear layer (batch_size, seq_len_q, seq_len_k, num_heads) @ (num_heads, 1)
        # -> (batch_size, seq_len_q, seq_len_k, 1) -> squeeze -> (batch_size, seq_len_q, seq_len_k)
        # We need to transpose attn_weights to (batch_size, seq_len_q, seq_len_k, num_heads) for linear layer
        interpreted_attn_weights = self.attn_interpret(attn_weights.permute(0, 2, 3, 1)).squeeze(-1) # Shape: (batch_size, seq_len_q, seq_len_k)


        return output, interpreted_attn_weights # Return output and interpreted attention weights

class TFTBinaryClassifier(pl.LightningModule):
    """
    あなたの既存 TFT(nn.Module) をラップして、
    ・forward でロジット返す
    ・training_step / validation_step / test_step で loss / metric 計算
    を行う LightningModule
    """
    def __init__(
        self,
        config: dict,
        lr: float = 2e-4,
        weight_decay: float = 1e-4,
        pos_weight: float = None,   # クラス不均衡が強ければ使用
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["config"])

        self.config = config
        self.model = TFT(config)   # 既存実装

        # 損失関数（ロジット入力）
        if pos_weight is not None:
            self.criterion = nn.BCEWithLogitsLoss(
                pos_weight=torch.tensor([pos_weight], dtype=torch.float32)
            )
        else:
            self.criterion = nn.BCEWithLogitsLoss()

    # ---- ここが一番重要：既存 TFT の forward と形を合わせる ----
    def forward(self, batch):
        """
        batch は CryptoBinaryDataset から返ってくる dict を想定。
        例:
            {
              "encoder_cont": (B, enc_len, n_cont_enc),
              "decoder_cont": (B, dec_len, n_cont_dec),
              "encoder_cat":  (B, enc_len, n_cat)  or None,
              "decoder_cat":  (B, dec_len, n_cat)  or None,
              ...
            }
        """
        # ★★ ここは「あなたの TFT.forward の引数」に合わせて書き換えてください ★★
        # 例1: TFT が x 全体を dict で受け取る場合
        # out = self.model(batch)

        # 例2: TFT が (enc_cont, dec_cont, enc_cat, dec_cat) を受け取る場合
        enc_cont = batch["encoder_cont"]
        dec_cont = batch["decoder_cont"]
        enc_cat = batch.get("encoder_cat", None)
        dec_cat = batch.get("decoder_cat", None)

        # あなたの tft.py の forward に合わせて ↓ を調整
        out = self.model(
            enc_cont,
            dec_cont,
            enc_cat=enc_cat,
            dec_cat=dec_cat,
        )

        # 戻り値の形も実装次第なので、よくあるパターンを吸収
        # 例A: dict で {"prediction_last_timestep": logits} を返す
        if isinstance(out, dict) and "prediction_last_timestep" in out:
            logits = out["prediction_last_timestep"]
        else:
            # 例B: (B, seq_len, 1) or (B, 1) or (B,)
            logits = out

        # 最後のタイムステップだけ使いたい場合
        if logits.dim() == 3:      # (B, T, 1)
            logits = logits[:, -1, :]
        if logits.dim() == 2 and logits.size(-1) == 1:  # (B, 1)
            logits = logits.squeeze(-1)                 # → (B,)

        return logits  # (B,)

    # ----------------- 共通 step -----------------
    def _shared_step(self, batch, stage: str):
        logits = self.forward(batch)      # (B,)
        targets = batch["target"].float() # {0,1}

        loss = self.criterion(logits, targets)

        with torch.no_grad():
            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).float()
            correct = (preds == targets).float().mean()

        self.log(f"{stage}_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log(f"{stage}_acc", correct, prog_bar=True, on_step=False, on_epoch=True)

        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        self._shared_step(batch, "val")

    def test_step(self, batch, batch_idx):
        self._shared_step(batch, "test")

    # ----------------- Optimizer -----------------
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        return optimizer

class TFT(nn.Module):
    def __init__(self, config):
        super(TFT, self).__init__()
        self.config = config
        self.static_variables = config.get("static_variables", 0)
        self.time_varying_real_variables_encoder = config.get("time_varying_real_variables_encoder", 0)
        self.time_varying_real_variables_decoder = config.get("time_varying_real_variables_decoder", 0) # Should be same as encoder
        self.time_varying_categoical_variables = config.get("time_varying_categoical_variables", 0)
        self.num_masked_series = config.get("num_masked_series", 0) # For future use if needed
        self.lstm_hidden_dimension = config.get("lstm_hidden_dimension", 64)
        self.lstm_layers = config.get("lstm_layers", 2)
        self.dropout = config.get("dropout", 0.3)
        self.embedding_dim = config.get("embedding_dim", 8)
        self.attn_heads = config.get("attn_heads", 4)
        self.num_quantiles = config.get("num_quantiles", 1) # For binary classification, typically 1
        self.valid_quantiles = config.get("valid_quantiles", [0.5]) # Not directly used in BCE loss
        self.seq_length = config.get("seq_length", 51) # Total sequence length (encoder + decoder)
        self.encode_length = config.get("encode_length", 50) # Encoder length
        self.predict_length = self.seq_length - self.encode_length # Decoder length

        # ★ 追加: 戦略数 & マルチヘッド使用フラグ
        self.num_strategies = config.get("num_strategies", 5)
        self.use_multi_head = config.get("use_multi_head", True)

        # Vocab sizes for embeddings
        self.static_embedding_vocab_sizes = config.get("static_embedding_vocab_sizes", [])
        self.time_varying_embedding_vocab_sizes = config.get("time_varying_embedding_vocab_sizes", [])


        # --- Input Transformations ---
        # Static features processing (assuming no static variables for now based on dataset)
        if self.static_variables > 0:
             # Real static features
             num_static_real = self.static_variables - len(self.static_embedding_vocab_sizes)
             if num_static_real > 0:
                  self.static_real_proj = nn.Linear(num_static_real, self.lstm_hidden_dimension)
             else:
                  self.static_real_proj = None

             # Categorical static features embeddings
             self.static_cat_embeddings = nn.ModuleList([
                 nn.Embedding(vocab_size, self.embedding_dim) for vocab_size in self.static_embedding_vocab_sizes
             ])
             # Gated Residual Network for static features
             static_grn_input_size = (self.lstm_hidden_dimension if num_static_real > 0 else 0) + len(self.static_embedding_vocab_sizes) * self.embedding_dim
             if static_grn_input_size > 0:
                  self.static_grn = GatedResidualNetwork(
                      input_size=static_grn_input_size,
                      hidden_size=self.lstm_hidden_dimension * 2, # Typically 2*hidden_size
                      output_size=self.lstm_hidden_dimension,
                      dropout=self.dropout
                  )
             else:
                  self.static_grn = None
        else:
             # Define dummy layers or None if no static features
             self.static_real_proj = None
             self.static_cat_embeddings = nn.ModuleList()
             self.static_grn = None


        # Time-varying real features transformation
        # Project real features to lstm_hidden_dimension
        if self.time_varying_real_variables_encoder > 0:
             self.time_varying_real_proj = nn.Linear(self.time_varying_real_variables_encoder, self.lstm_hidden_dimension)
        else:
             self.time_varying_real_proj = None # Or a dummy layer


        # Time-varying categorical features embeddings
        # Ensure vocab sizes are correct
        if len(self.time_varying_embedding_vocab_sizes) != self.time_varying_categoical_variables:
             print(f"Warning: Mismatch in time_varying_embedding_vocab_sizes length ({len(self.time_varying_embedding_vocab_sizes)}) and expected categorical variables ({self.time_varying_categoical_variables}). Initializing embeddings with dummy vocab sizes.")
             # Fallback to dummy embeddings if vocab sizes are inconsistent
             self.time_varying_cat_embeddings = nn.ModuleList([
                 nn.Embedding(2, self.embedding_dim) for _ in range(self.time_varying_categoical_variables)
             ])
        else:
             self.time_varying_cat_embeddings = nn.ModuleList([
                 nn.Embedding(vocab_size, self.embedding_dim) for vocab_size in self.time_varying_embedding_vocab_sizes
             ])

        # Gated Residual Network for time-varying features
        # Input size: lstm_hidden_dimension (from real proj) + sum of embedding dimensions (from cat embeddings)
        # Calculate actual embedding input size based on the successfully created embeddings
        actual_time_varying_embedding_input_size = sum(e.embedding_dim for e in self.time_varying_cat_embeddings)
        time_varying_grn_input_size = (self.lstm_hidden_dimension if self.time_varying_real_variables_encoder > 0 else 0) + actual_time_varying_embedding_input_size


        # Ensure GRN input size is positive before instantiating
        if time_varying_grn_input_size > 0:
             self.time_varying_grn = GatedResidualNetwork(
                 input_size=time_varying_grn_input_size,
                 hidden_size=self.lstm_hidden_dimension * 2,
                 output_size=self.lstm_hidden_dimension,
                 dropout=self.dropout
             )
        else:
             self.time_varying_grn = None # Or a dummy layer


        # --- LSTM Layer ---
        if self.lstm_hidden_dimension > 0:
             self.lstm = nn.LSTM(
                 input_size=self.lstm_hidden_dimension,
                 hidden_size=self.lstm_hidden_dimension,
                 num_layers=self.lstm_layers,
                 dropout=self.dropout if self.lstm_layers > 1 else 0,
                 batch_first=True
             )
        else:
             self.lstm = None


        # --- Gated Skip Connection ---
        if self.lstm_hidden_dimension > 0:
             self.gated_skip_grn = GatedResidualNetwork(
                 input_size=self.lstm_hidden_dimension,
                 hidden_size=self.lstm_hidden_dimension * 2,
                 output_size=self.lstm_hidden_dimension,
                 dropout=self.dropout
             )
        else:
             self.gated_skip_grn = None


        # --- Decoder ---
        if self.lstm_hidden_dimension > 0:
             self.self_attn = InterpretableMultiHeadAttention(
                 embed_dim=self.lstm_hidden_dimension,
                 num_heads=self.attn_heads,
                 dropout=self.dropout
             )
        else:
             self.self_attn = None

        if self.lstm_hidden_dimension > 0:
             self.post_attn_grn = GatedResidualNetwork(
                 input_size=self.lstm_hidden_dimension,
                 hidden_size=self.lstm_hidden_dimension * 2,
                 output_size=self.lstm_hidden_dimension,
                 dropout=self.dropout
             )
        else:
             self.post_attn_grn = None


        # --- Post-LSTM/Attention Processing ---
        if self.lstm_hidden_dimension > 0:
             self.final_grn = GatedResidualNetwork(
                 input_size=self.lstm_hidden_dimension,
                 hidden_size=self.lstm_hidden_dimension * 2,
                 output_size=self.lstm_hidden_dimension,
                 dropout=self.dropout
             )
        else:
             self.final_grn = None

        # ★ 共有 head（従来の出力層）
        if self.lstm_hidden_dimension > 0:
             self.output_layer = nn.Linear(self.lstm_hidden_dimension, self.num_quantiles)
        else:
             self.output_layer = None

        # ★ 戦略ごとの head（マルチヘッド）
        if self.lstm_hidden_dimension > 0 and self.use_multi_head:
            self.strategy_heads = nn.ModuleDict({
                str(i): nn.Linear(self.lstm_hidden_dimension, self.num_quantiles)
                for i in range(self.num_strategies)
            })
        else:
            self.strategy_heads = None


    def forward(
        self,
        x_enc_real,
        x_dec_real,
        x_enc_cat,
        x_dec_cat,
        x_enc_time_idx=None,
        x_dec_time_idx=None,
        static_cat_input=None,
        static_real_input=None,
        num_mask_series=0,
        strategy_ids=None,   # ★ 追加: 戦略ID (B,)  or None
    ):
        # x_enc_real: (batch_size, encode_len, num_real_features_encoder)
        # x_dec_real: (batch_size, predict_len, num_real_features_decoder)
        # x_enc_cat: (batch_size, encode_len, num_cat_features)
        # x_dec_cat: (batch_size, predict_len, num_cat_features)

        batch_size = x_enc_real.size(0)
        device = x_enc_real.device

        # --- Input Transformation ---
        if self.time_varying_real_proj is not None and x_enc_real.size(-1) > 0:
             x_enc_real_transformed = self.time_varying_real_proj(x_enc_real.float())
        else:
             x_enc_real_transformed = torch.zeros(batch_size, self.encode_length, self.lstm_hidden_dimension, device=device)

        if self.time_varying_real_proj is not None and x_dec_real.size(-1) > 0:
             x_dec_real_transformed = self.time_varying_real_proj(x_dec_real.float())
        else:
             x_dec_real_transformed = torch.zeros(batch_size, self.predict_length, self.lstm_hidden_dimension, device=device)

        # --- Categorical (encoder) ---
        x_enc_cat_embedded_list = []
        if len(self.time_varying_cat_embeddings) > 0 and x_enc_cat.size(-1) > 0:
             for i, embedding_layer in enumerate(self.time_varying_cat_embeddings):
                 clamped_input = torch.clamp(x_enc_cat[:, :, i].long(), 0, embedding_layer.num_embeddings - 1)
                 x_enc_cat_embedded_list.append(embedding_layer(clamped_input))
             if x_enc_cat_embedded_list:
                 x_enc_cat_embedded = torch.cat(x_enc_cat_embedded_list, dim=-1)
             else:
                 x_enc_cat_embedded = torch.empty(batch_size, self.encode_length, 0, device=device)
        else:
             x_enc_cat_embedded = torch.empty(batch_size, self.encode_length, 0, device=device)

        # --- Categorical (decoder) ---
        x_dec_cat_embedded_list = []
        if len(self.time_varying_cat_embeddings) > 0 and x_dec_cat.size(-1) > 0:
             for i, embedding_layer in enumerate(self.time_varying_cat_embeddings):
                  clamped_input = torch.clamp(x_dec_cat[:, :, i].long(), 0, embedding_layer.num_embeddings - 1)
                  x_dec_cat_embedded_list.append(embedding_layer(clamped_input))
             if x_dec_cat_embedded_list:
                 x_dec_cat_embedded = torch.cat(x_dec_cat_embedded_list, dim=-1)
             else:
                 x_dec_cat_embedded = torch.empty(batch_size, self.predict_length, 0, device=device)
        else:
             x_dec_cat_embedded = torch.empty(batch_size, self.predict_length, 0, device=device)

        # --- Combine Encoder ---
        inputs_enc_list = []
        if x_enc_real_transformed.size(-1) > 0:
             inputs_enc_list.append(x_enc_real_transformed)
        if x_enc_cat_embedded.size(-1) > 0:
             inputs_enc_list.append(x_enc_cat_embedded)

        if inputs_enc_list:
             x_enc_time_varying_grn_in = torch.cat(inputs_enc_list, dim=-1)
        else:
             if self.time_varying_grn is not None:
                  x_enc_time_varying_grn_in = torch.zeros(batch_size, self.encode_length, self.time_varying_grn.input_size, device=device)
             else:
                  x_enc_time_varying_grn_in = torch.empty(batch_size, self.encode_length, 0, device=device)

        # --- Combine Decoder ---
        inputs_dec_list = []
        if x_dec_real_transformed.size(-1) > 0:
             inputs_dec_list.append(x_dec_real_transformed)
        if x_dec_cat_embedded.size(-1) > 0:
             inputs_dec_list.append(x_dec_cat_embedded)

        if inputs_dec_list:
             x_dec_time_varying_grn_in = torch.cat(inputs_dec_list, dim=-1)
        else:
             if self.time_varying_grn is not None:
                  x_dec_time_varying_grn_in = torch.zeros(batch_size, self.predict_length, self.time_varying_grn.input_size, device=device)
             else:
                  x_dec_time_varying_grn_in = torch.empty(batch_size, self.predict_length, 0, device=device)

        # --- Time-varying GRN ---
        if self.time_varying_grn is not None:
             enc_grn_output = self.time_varying_grn(x_enc_time_varying_grn_in)
             dec_grn_output = self.time_varying_grn(x_dec_time_varying_grn_in)
        else:
             enc_grn_output = torch.zeros(batch_size, self.encode_length, self.lstm_hidden_dimension, device=device)
             dec_grn_output = torch.zeros(batch_size, self.predict_length, self.lstm_hidden_dimension, device=device)

        lstm_input = torch.cat([enc_grn_output, dec_grn_output], dim=1)

        # --- Static Features ---
        static_context = None
        if self.static_variables > 0 and static_cat_input is not None and static_real_input is not None:
            if self.static_real_proj is not None and static_real_input.size(-1) > 0:
                 static_real_transformed = self.static_real_proj(static_real_input.float())
            else:
                 static_real_transformed = torch.zeros(batch_size, self.lstm_hidden_dimension, device=device)

            static_cat_embedded_list = []
            if len(self.static_cat_embeddings) > 0 and static_cat_input.size(-1) > 0:
                 for i, embedding_layer in enumerate(self.static_cat_embeddings):
                     clamped_input = torch.clamp(static_cat_input[:, i].long(), 0, embedding_layer.num_embeddings - 1)
                     static_cat_embedded_list.append(embedding_layer(clamped_input))
                 if static_cat_embedded_list:
                      static_cat_embedded = torch.cat(static_cat_embedded_list, dim=-1)
                 else:
                      static_cat_embedded = torch.empty(batch_size, 0, device=device)
            else:
                static_cat_embedded = torch.empty(batch_size, 0, device=device)

            static_grn_input_list = []
            if static_real_transformed.size(-1) > 0:
                 static_grn_input_list.append(static_real_transformed)
            if static_cat_embedded.size(-1) > 0:
                 static_grn_input_list.append(static_cat_embedded)

            if self.static_grn is not None and static_grn_input_list:
                 static_grn_input = torch.cat(static_grn_input_list, dim=-1)
                 static_context = self.static_grn(static_grn_input)
                 static_context = static_context.unsqueeze(1).expand(-1, self.seq_length, -1)
            else:
                 static_context = torch.zeros(batch_size, self.seq_length, self.lstm_hidden_dimension, device=device)
        else:
             static_context = torch.zeros(batch_size, self.seq_length, self.lstm_hidden_dimension, device=device)

        lstm_input = lstm_input + static_context

        # --- LSTM ---
        lstm_output, (hn, cn) = self.lstm(lstm_input)

        # --- Gated Skip ---
        if self.gated_skip_grn is not None:
             lstm_output_gated_skip = self.gated_skip_grn(lstm_output)
        else:
             lstm_output_gated_skip = lstm_output

        # --- Self-Attention ---
        causal_mask = torch.triu(torch.ones(self.seq_length, self.seq_length), diagonal=1).bool().to(lstm_output_gated_skip.device)
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)
        causal_mask = ~causal_mask

        attn_input = lstm_output_gated_skip

        if self.self_attn is not None and attn_input.size(-1) > 0:
             attn_output, attn_weights = self.self_attn(attn_input, attn_input, attn_input, attn_mask=causal_mask)
        else:
             attn_output = torch.zeros_like(attn_input, device=device)
             attn_weights = torch.zeros(batch_size, self.seq_length, self.seq_length, device=device)

        # --- Post-attention GRN ---
        if self.post_attn_grn is not None:
             post_attn_output = self.post_attn_grn(attn_output)
        else:
             post_attn_output = attn_output

        # --- Final GRN ---
        if self.final_grn is not None:
             final_grn_output = self.final_grn(post_attn_output)
        else:
             final_grn_output = post_attn_output

        # --- Decoder hidden (encoder+decoderからdecoder部分だけ抽出) ---
        decoder_hidden = final_grn_output[:, self.encode_length:, :]  # (B, predict_len, hidden)

        # --- 出力部：shared head or strategy-specific head ---
        if self.strategy_heads is not None and strategy_ids is not None:
            # ★ マルチヘッドモード
            if not torch.is_tensor(strategy_ids):
                strategy_ids_tensor = torch.as_tensor(strategy_ids, device=device, dtype=torch.long)
            else:
                strategy_ids_tensor = strategy_ids.to(device=device, dtype=torch.long)

            # 最後のステップのみを head に通す
            h_last = decoder_hidden[:, -1, :]  # (B, hidden)
            logits_last = torch.zeros(batch_size, self.num_quantiles, device=device)

            unique_strats = torch.unique(strategy_ids_tensor)
            for s in unique_strats:
                mask = (strategy_ids_tensor == s)
                if mask.sum() == 0:
                    continue
                key = str(int(s.item()))
                if key in self.strategy_heads:
                    head = self.strategy_heads[key]
                else:
                    # 未定義戦略はとりあえず head "0" か shared head を使う
                    if "0" in self.strategy_heads:
                        head = self.strategy_heads["0"]
                    else:
                        head = self.output_layer
                h_s = h_last[mask]             # (B_s, hidden)
                logits_s = head(h_s)           # (B_s, num_quantiles)
                logits_last[mask] = logits_s

            # (B, predict_len, num_quantiles) 形式に戻す（最後のステップだけ値を入れる）
            decoder_output = torch.zeros(batch_size, self.predict_length, self.num_quantiles, device=device)
            decoder_output[:, -1, :] = logits_last

        else:
            # ★ 従来どおりの shared head
            if self.output_layer is not None and final_grn_output.size(-1) > 0:
                 decoder_output = self.output_layer(decoder_hidden)
            else:
                 decoder_output = torch.zeros(batch_size, self.predict_length, self.num_quantiles, device=device)

        # decoder_output: (B, predict_len, num_quantiles)
        return decoder_output, attn_weights


# class TFT(nn.Module):
#     def __init__(self, config):
#         super(TFT, self).__init__()
#         self.config = config
#         self.static_variables = config.get("static_variables", 0)
#         self.time_varying_real_variables_encoder = config.get("time_varying_real_variables_encoder", 0)
#         self.time_varying_real_variables_decoder = config.get("time_varying_real_variables_decoder", 0) # Should be same as encoder
#         self.time_varying_categoical_variables = config.get("time_varying_categoical_variables", 0)
#         self.num_masked_series = config.get("num_masked_series", 0) # For future use if needed
#         self.lstm_hidden_dimension = config.get("lstm_hidden_dimension", 64)
#         self.lstm_layers = config.get("lstm_layers", 2)
#         self.dropout = config.get("dropout", 0.3)
#         self.embedding_dim = config.get("embedding_dim", 8)
#         self.attn_heads = config.get("attn_heads", 4)
#         self.num_quantiles = config.get("num_quantiles", 1) # For binary classification, typically 1
#         self.valid_quantiles = config.get("valid_quantiles", [0.5]) # Not directly used in BCE loss
#         self.seq_length = config.get("seq_length", 51) # Total sequence length (encoder + decoder)
#         self.encode_length = config.get("encode_length", 50) # Encoder length
#         self.predict_length = self.seq_length - self.encode_length # Decoder length

#         # Vocab sizes for embeddings
#         self.static_embedding_vocab_sizes = config.get("static_embedding_vocab_sizes", [])
#         self.time_varying_embedding_vocab_sizes = config.get("time_varying_embedding_vocab_sizes", [])


#         # --- Input Transformations ---
#         # Static features processing (assuming no static variables for now based on dataset)
#         if self.static_variables > 0:
#              # Real static features
#              num_static_real = self.static_variables - len(self.static_embedding_vocab_sizes)
#              if num_static_real > 0:
#                   self.static_real_proj = nn.Linear(num_static_real, self.lstm_hidden_dimension)
#              else:
#                   self.static_real_proj = None

#              # Categorical static features embeddings
#              self.static_cat_embeddings = nn.ModuleList([
#                  nn.Embedding(vocab_size, self.embedding_dim) for vocab_size in self.static_embedding_vocab_sizes
#              ])
#              # Gated Residual Network for static features
#              static_grn_input_size = (self.lstm_hidden_dimension if num_static_real > 0 else 0) + len(self.static_embedding_vocab_sizes) * self.embedding_dim
#              if static_grn_input_size > 0:
#                   self.static_grn = GatedResidualNetwork(
#                       input_size=static_grn_input_size,
#                       hidden_size=self.lstm_hidden_dimension * 2, # Typically 2*hidden_size
#                       output_size=self.lstm_hidden_dimension,
#                       dropout=self.dropout
#                   )
#              else:
#                   self.static_grn = None
#         else:
#              # Define dummy layers or None if no static features
#              self.static_real_proj = None
#              self.static_cat_embeddings = nn.ModuleList()
#              self.static_grn = None


#         # Time-varying real features transformation
#         # Project real features to lstm_hidden_dimension
#         if self.time_varying_real_variables_encoder > 0:
#              self.time_varying_real_proj = nn.Linear(self.time_varying_real_variables_encoder, self.lstm_hidden_dimension)
#         else:
#              self.time_varying_real_proj = None # Or a dummy layer


#         # Time-varying categorical features embeddings
#         # Ensure vocab sizes are correct
#         if len(self.time_varying_embedding_vocab_sizes) != self.time_varying_categoical_variables:
#              print(f"Warning: Mismatch in time_varying_embedding_vocab_sizes length ({len(self.time_varying_embedding_vocab_sizes)}) and expected categorical variables ({self.time_varying_categoical_variables}). Initializing embeddings with dummy vocab sizes.")
#              # Fallback to dummy embeddings if vocab sizes are inconsistent
#              self.time_varying_cat_embeddings = nn.ModuleList([
#                  nn.Embedding(2, self.embedding_dim) for _ in range(self.time_varying_categoical_variables)
#              ])
#         else:
#              self.time_varying_cat_embeddings = nn.ModuleList([
#                  nn.Embedding(vocab_size, self.embedding_dim) for vocab_size in self.time_varying_embedding_vocab_sizes
#              ])

#         # Gated Residual Network for time-varying features
#         # Input size: lstm_hidden_dimension (from real proj) + sum of embedding dimensions (from cat embeddings)
#         # Calculate actual embedding input size based on the successfully created embeddings
#         actual_time_varying_embedding_input_size = sum(e.embedding_dim for e in self.time_varying_cat_embeddings)
#         time_varying_grn_input_size = (self.lstm_hidden_dimension if self.time_varying_real_variables_encoder > 0 else 0) + actual_time_varying_embedding_input_size


#         # Ensure GRN input size is positive before instantiating
#         if time_varying_grn_input_size > 0:
#              self.time_varying_grn = GatedResidualNetwork(
#                  input_size=time_varying_grn_input_size,
#                  hidden_size=self.lstm_hidden_dimension * 2,
#                  output_size=self.lstm_hidden_dimension,
#                  dropout=self.dropout
#              )
#         else:
#              self.time_varying_grn = None # Or a dummy layer


#         # --- LSTM Layer ---
#         # LSTM input size is always lstm_hidden_dimension as it takes output from time_varying_grn
#         # Check if lstm_hidden_dimension is positive before instantiating LSTM
#         if self.lstm_hidden_dimension > 0:
#              self.lstm = nn.LSTM(
#                  input_size=self.lstm_hidden_dimension,
#                  hidden_size=self.lstm_hidden_dimension,
#                  num_layers=self.lstm_layers,
#                  dropout=self.dropout if self.lstm_layers > 1 else 0, # Dropout only if num_layers > 1
#                  batch_first=True # Input Tensors are (batch_size, seq_len, input_size)
#              )
#         else:
#              self.lstm = None # Or a dummy LSTM


#         # --- Gated Skip Connection ---
#         # Check if lstm_hidden_dimension is positive before instantiating GRN
#         if self.lstm_hidden_dimension > 0:
#              self.gated_skip_grn = GatedResidualNetwork(
#                  input_size=self.lstm_hidden_dimension, # LSTM output size
#                  hidden_size=self.lstm_hidden_dimension * 2,
#                  output_size=self.lstm_hidden_dimension,
#                  dropout=self.dropout
#              )
#         else:
#              self.gated_skip_grn = None


#         # --- Decoder ---
#         # Self-Attention layer
#         # Check if lstm_hidden_dimension is positive before instantiating Attention
#         if self.lstm_hidden_dimension > 0:
#              self.self_attn = InterpretableMultiHeadAttention(
#                  embed_dim=self.lstm_hidden_dimension,
#                  num_heads=self.attn_heads,
#                  dropout=self.dropout
#              )
#         else:
#              self.self_attn = None

#         # Gated Residual Network after Self-Attention
#         # Check if lstm_hidden_dimension is positive before instantiating GRN
#         if self.lstm_hidden_dimension > 0:
#              self.post_attn_grn = GatedResidualNetwork(
#                  input_size=self.lstm_hidden_dimension,
#                  hidden_size=self.lstm_hidden_dimension * 2,
#                  output_size=self.lstm_hidden_dimension,
#                  dropout=self.dropout
#              )
#         else:
#              self.post_attn_grn = None


#         # --- Post-LSTM/Attention Processing ---
#         # Gated Residual Network for final prediction input
#         # Check if lstm_hidden_dimension is positive before instantiating GRN
#         if self.lstm_hidden_dimension > 0:
#              self.final_grn = GatedResidualNetwork(
#                  input_size=self.lstm_hidden_dimension, # Input from post_attn_grn
#                  hidden_size=self.lstm_hidden_dimension * 2,
#                  output_size=self.lstm_hidden_dimension,
#                  dropout=self.dropout
#              )
#         else:
#              self.final_grn = None

#         # Output layer (for binary classification, predict a single value per quantile)
#         # Check if lstm_hidden_dimension is positive before instantiating Output Layer
#         if self.lstm_hidden_dimension > 0:
#              self.output_layer = nn.Linear(self.lstm_hidden_dimension, self.num_quantiles)
#         else:
#              self.output_layer = None


#     # Modified forward method to accept split inputs from CryptoBinaryDataset
#     # Added checks for empty tensors (dimension 0)
#     # Added debug prints for all categorical features
#     # Added clamping for categorical inputs as a safeguard
#     def forward(self, x_enc_real, x_dec_real, x_enc_cat, x_dec_cat, x_enc_time_idx=None, x_dec_time_idx=None, static_cat_input=None, static_real_input=None, num_mask_series=0):
#         # x_enc_real: (batch_size, encode_len, num_real_features_encoder)
#         # x_dec_real: (batch_size, predict_len, num_real_features_decoder)
#         # x_enc_cat: (batch_size, encode_len, num_cat_features) - Expected as Long type for embedding indices
#         # x_dec_cat: (batch_size, predict_len, num_cat_features) - Expected as Long type for embedding indices
#         # x_enc_time_idx: (batch_size, encode_len) - Optional
#         # x_dec_time_idx: (batch_size, predict_len) - Optional
#         # static_cat_input: (batch_size, num_static_cat) - Optional
#         # static_real_input: (batch_size, num_static_real) - Optional
#         # num_mask_series: int - Optional

#         # Determine batch size and device from a guaranteed input tensor
#         # Use x_enc_real as the reference for batch_size and device
#         batch_size = x_enc_real.size(0)
#         device = x_enc_real.device


#         # --- Input Transformation ---
#         # Handle time-varying real features (Encoder and Decoder)
#         # Check if the linear layer exists and if the input has features
#         if self.time_varying_real_proj is not None and x_enc_real.size(-1) > 0:
#              x_enc_real_transformed = self.time_varying_real_proj(x_enc_real.float()) # Ensure float type
#         else:
#              x_enc_real_transformed = torch.zeros(batch_size, self.encode_length, self.lstm_hidden_dimension, device=device) # Create dummy if no real features


#         # Decoder real input transformation
#         if self.time_varying_real_proj is not None and x_dec_real.size(-1) > 0:
#              x_dec_real_transformed = self.time_varying_real_proj(x_dec_real.float()) # Ensure float type
#         else:
#              x_dec_real_transformed = torch.zeros(batch_size, self.predict_length, self.lstm_hidden_dimension, device=device) # Create dummy if no real features


#         # Handle time-varying categorical features (Encoder and Decoder)
#         # Embed each categorical feature and concatenate embeddings
#         x_enc_cat_embedded_list = []
#         # Check if there are categorical embedding layers and if the input has categorical features
#         if len(self.time_varying_cat_embeddings) > 0 and x_enc_cat.size(-1) > 0:
#             #  # Debugging: Check categorical input ranges before embedding for ALL features
#             #  if self.config.get("debug_categorical_input", False):
#             #       print(f"Debug: x_enc_cat shape: {x_enc_cat.shape}")
#             #       print(f"Debug: x_enc_cat dtype: {x_enc_cat.dtype}")
#             #       # print(f"Debug: x_enc_cat unique values for all {x_enc_cat.size(-1)} features:") # Moved this print below
#             #       for i in range(x_enc_cat.size(-1)): # Loop through all features
#             #            try:
#             #                 # Ensure data is Long before min/max/unique and check
#             #                 feature_data = x_enc_cat[:, :, i].long()
#             #                 unique_vals = torch.unique(feature_data)
#             #                 min_val = torch.min(feature_data)
#             #                 max_val = torch.max(feature_data)
#             #                 vocab_size = self.time_varying_embedding_vocab_sizes[i] if i < len(self.time_varying_embedding_vocab_sizes) else -1 # Get expected vocab size

#             #                 # Check for out-of-bounds values
#             #                 is_out_of_bounds = False
#             #                 # Allow index 0 even if vocab_size is 1 (single class)
#             #                 if vocab_size > 0 and (min_val < 0 or max_val >= vocab_size):
#             #                     is_out_of_bounds = True
#             #                     print(f"  !!! Potential OOB for Encoder Feature {i} (Expected Vocab: {vocab_size}) - Min: {min_val}, Max: {max_val}, Unique: {unique_vals}")
#             #                 else:
#             #                     print(f"  Encoder Feature {i} (Expected Vocab: {vocab_size}) - Min: {min_val}, Max: {max_val}, Unique: {unique_vals}")

#             #            except Exception as e:
#             #                 print(f"  Error debugging Encoder Feature {i}: {e}")


#              for i, embedding_layer in enumerate(self.time_varying_cat_embeddings):
#                  # Ensure input indices are within the valid range for the embedding layer
#                  # If using CryptoBinaryDataset, this should already be handled by correct DataFrame dtype (int64)
#                  # and valid vocab sizes determined during dataset creation.
#                  # Add a clamping step as a safeguard, but ideally data preparation should ensure valid indices.
#                  clamped_input = torch.clamp(x_enc_cat[:, :, i].long(), 0, embedding_layer.num_embeddings - 1)
#                  x_enc_cat_embedded_list.append(embedding_layer(clamped_input))

#              if x_enc_cat_embedded_list: # Check if the list is not empty before concatenating
#                  x_enc_cat_embedded = torch.cat(x_enc_cat_embedded_list, dim=-1) # (batch_size, encode_len, total_embedding_dim)
#              else:
#                  x_enc_cat_embedded = torch.empty(batch_size, self.encode_length, 0, device=device) # Empty if no cat features

#         else:
#              x_enc_cat_embedded = torch.empty(batch_size, self.encode_length, 0, device=device) # Empty if no cat features


#         x_dec_cat_embedded_list = []
#         # Check if there are categorical embedding layers and if the input has categorical features
#         if len(self.time_varying_cat_embeddings) > 0 and x_dec_cat.size(-1) > 0:
#             #  # Debugging: Check categorical input ranges before embedding for ALL features
#             #  if self.config.get("debug_categorical_input", False):
#             #       print(f"Debug: x_dec_cat shape: {x_dec_cat.shape}")
#             #       print(f"Debug: x_dec_cat dtype: {x_dec_cat.dtype}")
#             #       # print(f"Debug: x_dec_cat unique values for all {x_dec_cat.size(-1)} features:") # Moved this print below
#             #       for i in range(x_dec_cat.size(-1)): # Loop through all features
#             #            try:
#             #                 # Ensure data is Long before min/max/unique and check
#             #                 feature_data = x_dec_cat[:, :, i].long()
#             #                 unique_vals = torch.unique(feature_data)
#             #                 min_val = torch.min(feature_data)
#             #                 max_val = torch.max(feature_data)
#             #                 vocab_size = self.time_varying_embedding_vocab_sizes[i] if i < len(self.time_varying_embedding_vocab_sizes) else -1 # Get expected vocab size

#             #                  # Check for out-of-bounds values
#             #                 is_out_of_bounds = False
#             #                 # Allow index 0 even if vocab_size is 1 (single class)
#             #                 if vocab_size > 0 and (min_val < 0 or max_val >= vocab_size):
#             #                     is_out_of_bounds = True
#             #                     print(f"  !!! Potential OOB for Decoder Feature {i} (Expected Vocab: {vocab_size}) - Min: {min_val}, Max: {max_val}, Unique: {unique_vals}")
#             #                 else:
#             #                     print(f"  Decoder Feature {i} (Expected Vocab: {vocab_size}) - Min: {min_val}, Max: {max_val}, Unique: {unique_vals}")

#             #            except Exception as e:
#             #                 print(f"  Error debugging Decoder Feature {i}: {e}")


#              for i, embedding_layer in enumerate(self.time_varying_cat_embeddings):
#                   clamped_input = torch.clamp(x_dec_cat[:, :, i].long(), 0, embedding_layer.num_embeddings - 1)
#                   x_dec_cat_embedded_list.append(embedding_layer(clamped_input))
#              if x_dec_cat_embedded_list: # Check if the list is not empty before concatenating
#                  x_dec_cat_embedded = torch.cat(x_dec_cat_embedded_list, dim=-1) # (batch_size, predict_len, total_embedding_dim)
#              else:
#                  x_dec_cat_embedded = torch.empty(batch_size, self.predict_length, 0, device=device) # Empty if no cat features

#         else:
#              x_dec_cat_embedded = torch.empty(batch_size, self.predict_length, 0, device=device) # Empty if no cat features


#         # Combine real and categorical embeddings/transformations
#         # Encoder input to time_varying_grn
#         inputs_enc_list = []
#         if x_enc_real_transformed.size(-1) > 0:
#              inputs_enc_list.append(x_enc_real_transformed)
#         if x_enc_cat_embedded.size(-1) > 0:
#              inputs_enc_list.append(x_enc_cat_embedded)

#         if inputs_enc_list:
#              x_enc_time_varying_grn_in = torch.cat(inputs_enc_list, dim=-1) # (batch_size, encode_len, input_size_grn)
#         else:
#              # Handle case where both real and categorical inputs are empty for encoder
#              # Create a dummy tensor with the expected input size for the GRN, filled with zeros
#              # The input_size for time_varying_grn is calculated in __init__
#              if self.time_varying_grn is not None:
#                   x_enc_time_varying_grn_in = torch.zeros(batch_size, self.encode_length, self.time_varying_grn.input_size, device=device)
#              else: # If time_varying_grn is None (input_size was 0)
#                   x_enc_time_varying_grn_in = torch.empty(batch_size, self.encode_length, 0, device=device) # Empty tensor


#         # Decoder input to time_varying_grn
#         inputs_dec_list = []
#         if x_dec_real_transformed.size(-1) > 0:
#              inputs_dec_list.append(x_dec_real_transformed)
#         if x_dec_cat_embedded.size(-1) > 0:
#              inputs_dec_list.append(x_dec_cat_embedded)

#         if inputs_dec_list:
#              x_dec_time_varying_grn_in = torch.cat(inputs_dec_list, dim=-1) # (batch_size, predict_len, input_size_grn)
#         else:
#              # Handle case where both real and categorical inputs are empty for decoder
#              if self.time_varying_grn is not None:
#                   x_dec_time_varying_grn_in = torch.zeros(batch_size, self.predict_length, self.time_varying_grn.input_size, device=device)
#              else:
#                   x_dec_time_varying_grn_in = torch.empty(batch_size, self.predict_length, 0, device=device) # Empty tensor


#         # Apply time-varying GRN
#         # Note: This GRN does not take additional_input in the paper's standard form
#         # unless it's for time_steps (which we handle differently).
#         # If time_idx is needed, it would be concatenated here or handled as static.
#         # Assuming time_idx is NOT included in the input tensors x_enc_real/cat etc.
#         # and is handled internally by the model if needed (e.g., relative time index).

#         # Check if the GRN layer exists before calling it
#         if self.time_varying_grn is not None:
#              enc_grn_output = self.time_varying_grn(x_enc_time_varying_grn_in) # (batch_size, encode_len, hidden_size)
#              dec_grn_output = self.time_varying_grn(x_dec_time_varying_grn_in) # (batch_size, predict_len, hidden_size)
#         else: # If no time-varying features, GRN output is just zeros
#              enc_grn_output = torch.zeros(batch_size, self.encode_length, self.lstm_hidden_dimension, device=device)
#              dec_grn_output = torch.zeros(batch_size, self.predict_length, self.lstm_hidden_dimension, device=device)


#         # Concatenate encoder and decoder GRN outputs for LSTM
#         # Shape: (batch_size, seq_length, hidden_size)
#         lstm_input = torch.cat([enc_grn_output, dec_grn_output], dim=1)

#         # --- Static Features Processing (if any) ---
#         static_context = None
#         if self.static_variables > 0 and static_cat_input is not None and static_real_input is not None:
#             # Process static real features (check if layer exists and input has features)
#             static_real_transformed = None
#             if self.static_real_proj is not None and static_real_input.size(-1) > 0:
#                  static_real_transformed = self.static_real_proj(static_real_input.float()) # Ensure float type # (batch_size, hidden_size)
#             else:
#                  static_real_transformed = torch.zeros(batch_size, self.lstm_hidden_dimension, device=device) # Dummy if no real static

#             # Process static categorical features (check if embeddings exist and input has features)
#             static_cat_embedded_list = []
#             if len(self.static_cat_embeddings) > 0 and static_cat_input.size(-1) > 0:
#                  for i, embedding_layer in enumerate(self.static_cat_embeddings):
#                      clamped_input = torch.clamp(static_cat_input[:, i].long(), 0, embedding_layer.num_embeddings - 1)
#                      static_cat_embedded_list.append(embedding_layer(clamped_input)) # Ensure long type # (batch_size, embedding_dim)

#                  if static_cat_embedded_list: # Check if the list is not empty
#                       static_cat_embedded = torch.cat(static_cat_embedded_list, dim=-1) # (batch_size, total_static_embedding_dim)
#                  else:
#                       static_cat_embedded = torch.empty(batch_size, 0, device=device) # Empty if no cat static
#             else:
#                 static_cat_embedded = torch.empty(batch_size, 0, device=device) # Empty if no cat static


#             # Combine static features for static GRN (check if GRN exists and input size is correct)
#             static_grn_input_list = []
#             if static_real_transformed.size(-1) > 0:
#                  static_grn_input_list.append(static_real_transformed)
#             if static_cat_embedded.size(-1) > 0:
#                  static_grn_input_list.append(static_cat_embedded)

#             if self.static_grn is not None and static_grn_input_list:
#                  static_grn_input = torch.cat(static_grn_input_list, dim=-1)
#                  static_context = self.static_grn(static_grn_input) # (batch_size, hidden_size)
#                  # Expand static_context to match sequence length for addition to LSTM output
#                  static_context = static_context.unsqueeze(1).expand(-1, self.seq_length, -1) # (batch_size, seq_length, hidden_size)
#             else:
#                  static_context = torch.zeros(batch_size, self.seq_length, self.lstm_hidden_dimension, device=device) # Dummy if no static features or GRN
#         else:
#              # If no static variables config, static_context is zero tensor
#              static_context = torch.zeros(batch_size, self.seq_length, self.lstm_hidden_dimension, device=device)


#         # Add static context to LSTM input
#         lstm_input = lstm_input + static_context

#         # Initial LSTM state (usually zeros if not using static features for initialization)
#         # h0 = torch.zeros(self.lstm_layers, batch_size, self.lstm_hidden_dimension).to(lstm_input.device) # Use lstm_input.device
#         # c0 = torch.zeros(self.lstm_layers, batch_size, self.lstm_hidden_dimension).to(lstm_input.device) # Use lstm_input.device

#         # LSTM forward pass
#         # output: (batch_size, seq_length, hidden_size)
#         # hn, cn: final hidden/cell states (num_layers, batch_size, hidden_size)
#         # No initial hidden/cell state passed explicitly, LSTM defaults to zeros
#         lstm_output, (hn, cn) = self.lstm(lstm_input)


#         # --- Gated Skip Connection ---
#         # This connects the LSTM output to the post-attention GRN input
#         # The skip connection is applied *before* the self-attention
#         # Check if the gated skip GRN layer exists before calling it
#         if self.gated_skip_grn is not None:
#              lstm_output_gated_skip = self.gated_skip_grn(lstm_output) # (batch_size, seq_length, hidden_size)
#         else:
#              # If no gated skip GRN (input size was 0, should not happen here), use LSTM output directly or a zero tensor
#              # Based on typical TFT architecture, input size should be lstm_hidden_dimension > 0 if lstm_hidden_dimension > 0
#              lstm_output_gated_skip = lstm_output # Fallback


#         # --- Decoder: Self-Attention ---
#         # Apply self-attention over the entire sequence (encoder + decoder parts)
#         # Query, Key, Value are all from the LSTM output (after gated skip)
#         # We need a causal mask to prevent attending to future timesteps in the decoder part.
#         # The attention mask should be (batch_size, 1, seq_length, seq_length) or (seq_length, seq_length)
#         # For causal attention, element (i, j) of the mask is 0 if j > i, and 1 otherwise.
#         causal_mask = torch.triu(torch.ones(self.seq_length, self.seq_length), diagonal=1).bool().to(lstm_output_gated_skip.device)
#         causal_mask = causal_mask.unsqueeze(0).unsqueeze(0) # Shape: (1, 1, seq_length, seq_length)
#         # Invert the mask for masked_fill (masked_fill masks where mask is True)
#         causal_mask = ~causal_mask # Shape: (1, 1, seq_length, seq_length), True where attending is allowed

#         attn_input = lstm_output_gated_skip # (batch_size, seq_length, hidden_size)

#         # Apply self-attention (check if layer exists and input has features)
#         if self.self_attn is not None and attn_input.size(-1) > 0:
#              # attn_output: (batch_size, seq_length, hidden_size)
#              # attn_weights: (batch_size, seq_length, seq_length) - Interpreted attention weights
#              attn_output, attn_weights = self.self_attn(attn_input, attn_input, attn_input, attn_mask=causal_mask)
#         else:
#              # If no self-attention layer or input has no features, return zeros and dummy attention weights
#              attn_output = torch.zeros_like(attn_input, device=device) # Return zero tensor with same shape as input
#              attn_weights = torch.zeros(batch_size, self.seq_length, self.seq_length, device=device) # Dummy attention weights


#         # --- Gated Residual Network after Self-Attention ---
#         # Input to this GRN is the output of the self-attention
#         # Check if the post-attention GRN layer exists before calling it
#         if self.post_attn_grn is not None:
#              post_attn_output = self.post_attn_grn(attn_output) # (batch_size, seq_length, hidden_size)
#         else:
#              # If no post-attention GRN (input size was 0, should not happen if hidden_dimension > 0), use attn_output directly
#              post_attn_output = attn_output # Fallback


#         # --- Final Processing and Output ---
#         # The final GRN takes the output of the post-attention GRN
#         # Check if the final GRN layer exists before calling it
#         if self.final_grn is not None:
#              final_grn_output = self.final_grn(post_attn_output) # (batch_size, seq_length, hidden_size)
#         else:
#              # If no final GRN (input size was 0), use post_attn_output directly
#              final_grn_output = post_attn_output # Fallback


#         # Pass the final GRN output through the output layer
#         # We are interested in the predictions for the decoder part (future)
#         # Slice the output to get only the decoder timesteps
#         # Shape: (batch_size, predict_len, hidden_size) -> (batch_size, predict_len, num_quantiles)
#         # Check if the output layer exists before calling it
#         if self.output_layer is not None and final_grn_output.size(-1) > 0:
#              decoder_output = self.output_layer(final_grn_output[:, self.encode_length:, :])
#         else:
#              # If no output layer or input has no features, return zero tensor of the correct output shape
#              decoder_output = torch.zeros(batch_size, self.predict_length, self.num_quantiles, device=device)


#         # For binary classification, typically num_quantiles=1.
#         # The output is logits for the binary outcome.
#         # Shape: (batch_size, predict_len, 1)

#         # The loss calculation in the training loop should use BCEWithLoss
#         # and compare the prediction for the LAST timestep of the decoder
#         # (decoder_output[:, -1, 0]) with the single binary target per sequence.

#         # Return the decoder output and potentially attention weights for interpretability
#         return decoder_output, attn_weights # Return decoder output and attention weights


