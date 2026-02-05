from transformers import BertForSequenceClassification, BertConfig
import torch
import torch.nn as nn
import torch.nn.functional as F


class TextFeatureExtractor(nn.Module):
    """BERT-based text feature extractor"""

    def __init__(self, pretrained_name: str, num_labels: int, use_selfclassifier_flag=False, dropout_prob: float = 0.3,
                 start_freeze_layer=11):
        super().__init__()
        self.config = BertConfig.from_pretrained(
            pretrained_name,
            hidden_dropout_prob=dropout_prob,
            attention_probs_dropout_prob=dropout_prob
        )
        self.bert = BertForSequenceClassification.from_pretrained(
            pretrained_name,
            num_labels=num_labels,
            output_hidden_states=True,
            ignore_mismatched_sizes=True
        )
        self.warmup = False

        # Whether to use a custom classifier head (e.g., the self.classifier defined below)
        self.use_selfclassifier_flag = use_selfclassifier_flag

        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout_prob)

        # Custom classification head
        self.classifier = nn.Sequential(
            nn.Linear(768, 768),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(768, num_labels)
        )

        # Freeze the first several BERT encoder layers by setting requires_grad=False
        for param in self.bert.bert.encoder.layer[:start_freeze_layer].parameters():
            param.requires_grad = False

    def forward(self, input_ids, attention_mask):
        # Forward pass for each input text
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )

        # Extract the [CLS] token representation from the last hidden layer
        cls_output = outputs.hidden_states[-1][:, 0, :]  # (batch_size, hidden_size)

        if self.warmup:
            if self.use_selfclassifier_flag:
                logits = self.classifier(self.dropout(cls_output))
                return self.softmax(logits)
            else:
                logits = outputs.logits
                return self.softmax(logits)
        else:
            return cls_output



class UnifiedFlowModel(nn.Module):
    def __init__(self, stat_feat_dim, seq_feat_dim=28, seq_len=100, hidden_dim=128, num_classes=100):
        super().__init__()

        # FlowTransformerClassifier branch (statistical features)
        self.stat_embedding = nn.Linear(stat_feat_dim, hidden_dim)
        self.stat_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=4,
                dropout=0.1,
                batch_first=True
            ),
            num_layers=2
        )

        # Positional encoding to provide temporal order information to the Transformer
        self.positional_encoding = nn.Parameter(torch.zeros(1, 1000, hidden_dim))  # maximum sequence length = 1000
        nn.init.trunc_normal_(self.positional_encoding, std=0.02)

        # DualTransformer branch (packet-level feature sequences)
        self.feat_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=seq_feat_dim,
                nhead=4,
                dropout=0.1,
                batch_first=True
            ),
            num_layers=2
        )

        self.proj = nn.Linear(seq_feat_dim, hidden_dim)

        self.seq_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=8,
                dropout=0.1,
                batch_first=True
            ),
            num_layers=4
        )

        self.gate_heads = nn.ModuleList(
            [nn.Linear(hidden_dim * 2, 1) for _ in range(3)]
        )

        # Fusion layers and classifier
        self.dropout = nn.Dropout(0.5)

        self.fusion_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.7),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, num_classes)
        )

    def normalize_seq_x(self, seq_x):
        # Per-sample normalization (preserving temporal independence)
        mean = seq_x.mean(dim=1, keepdim=True)  # [B, 1, F]
        std = seq_x.std(dim=1, keepdim=True)    # [B, 1, F]
        return (seq_x - mean) / (std + 1e-6)

    def forward(self, stat_x, seq_x):
        seq_x = self.normalize_seq_x(seq_x)

        # stat_x: (B, 1, stat_feat_dim), statistical features
        # seq_x:  (B, T, seq_feat_dim=28), packet-level feature sequence

        # --- Statistical feature encoding ---
        stat_feat = self.stat_embedding(stat_x)
        stat_feat = self.stat_encoder(stat_feat)
        stat_feat = stat_feat.mean(dim=1)  # [B, hidden_dim]

        # --- Sequential feature encoding ---
        B, T, F = seq_x.shape
        seq_x = seq_x.reshape(B * T, F)
        seq_x = self.feat_encoder(seq_x)
        seq_x = seq_x.reshape(B, T, F)

        seq_feat = self.proj(seq_x)
        seq_feat = seq_feat + self.positional_encoding[:, :seq_feat.size(1), :]  # add positional encoding
        seq_feat = self.seq_encoder(seq_feat)
        seq_feat = seq_feat.mean(dim=1)  # [B, hidden_dim]

        # --- Feature fusion ---
        fusion = torch.cat([stat_feat, seq_feat], dim=1)  # [B, hidden_dim * 2]
        fusion = self.fusion_proj(fusion)

        if self.warmup:
            out = self.classifier(fusion)
            return out
        else:
            return fusion



class FusionModel(nn.Module):
    def __init__(self, hidden_size=64, trans_hidden_size=64, num_classes=10, dropout=0.3):
        super(FusionModel, self).__init__()
        self.trans_hidden_size = trans_hidden_size
        self.hidden_size = hidden_size

        # Learnable temperature parameter for adaptive scaling
        self.temperature = nn.Parameter(torch.tensor(0.5))

        # Projection layer for CLS embedding dimensionality reduction
        self.cls_proj = nn.Linear(768, trans_hidden_size)
        self.trans_proj = nn.Linear(trans_hidden_size, trans_hidden_size)

        # Layer normalization for BERT and Transformer features
        self.norm1 = nn.LayerNorm(self.trans_hidden_size)  # BERT features
        self.norm2 = nn.LayerNorm(self.trans_hidden_size)  # Transformer features

        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.pre_fusion_dropout = nn.Dropout(dropout)

        # Multiple gating heads to reduce single-point failure
        self.gate_heads = nn.ModuleList(
            [nn.Linear(trans_hidden_size * 2, 1) for _ in range(3)]
        )

        self.gate_proj = nn.Sequential(
            nn.Linear(trans_hidden_size, trans_hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(trans_hidden_size, trans_hidden_size)
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(trans_hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes)
        )

    def forward(self, bert_hidden, trans_hidden, labels=None):
        """
        bert_hidden: [B, H]  - BERT hidden representation (e.g., CLS embedding)
        trans_hidden: [B, H] - Transformer hidden representation (sequence-level features)
        """

        bert_hidden = self.relu(self.dropout(self.cls_proj(bert_hidden)))
        bert_hidden = self.norm1(bert_hidden)

        trans_hidden = self.relu(self.dropout(trans_hidden))
        trans_hidden = self.norm2(trans_hidden)

        combined = torch.cat((bert_hidden, trans_hidden), dim=1)
        combined = self.pre_fusion_dropout(combined)

        # Each gate head produces an independent gating weight
        gate_logits = torch.stack([head(combined) for head in self.gate_heads], dim=-1)
        gate = torch.softmax(gate_logits, dim=-1)  # [B, 1, 3]

        # Select the maximum gate value across heads
        max_gate, _ = gate.max(dim=-1)

        fused_feats = max_gate * bert_hidden + (1 - max_gate) * trans_hidden
        fused_feats = self.gate_proj(fused_feats)

        logits = self.classifier(fused_feats)

        if labels is not None:
            loss = F.cross_entropy(logits, labels)
            return loss, logits
        else:
            return logits
