"""Paper topology: Krause et al., EUSIPCO 2024, arXiv:2403.11827v2.

This is a fresh implementation, not the authors' unpublished training code.
Unspecified hyperparameters and the paper's pooling inconsistency are recorded
in docs/paper_reproduction.md. The default preserves the noncausal architecture.
"""
from dataclasses import dataclass, asdict
import torch
from torch import nn


@dataclass
class ModelConfig:
    classes: int = 13
    input_channels: int = 4
    frequency_bins: int = 512
    cnn_channels: int = 128
    frequency_pool: tuple = (8, 8, 4)
    time_pool: tuple = (5, 1, 1)
    gru_hidden: int = 128
    gru_layers: int = 2
    attention_layers: int = 2
    attention_heads: int = 8
    head_hidden: int = 128
    dropout: float = .05
    output_mode: str = 'multi_accddoa'
    tracks: int = 3
    self_classes: int = 0

    def to_dict(self):
        return asdict(self)


class PaperSELD(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        if cfg.output_mode not in ['multi_accddoa', 'multi_task']:
            raise ValueError('Unknown paper output method')
        if cfg.tracks != 3:
            raise ValueError('The implemented ADPIT representation uses three tracks')
        self.cfg = cfg
        layers, channels, freq = [], cfg.input_channels, cfg.frequency_bins
        for fp, tp in zip(cfg.frequency_pool, cfg.time_pool):
            layers.append(nn.Sequential(nn.Conv2d(channels, cfg.cnn_channels, 3, padding=1),
                nn.BatchNorm2d(cfg.cnn_channels), nn.ReLU(), nn.MaxPool2d((tp, fp)), nn.Dropout2d(cfg.dropout)))
            channels, freq = cfg.cnn_channels, freq // fp
        if freq < 1 or cfg.gru_hidden % cfg.attention_heads:
            raise ValueError('Incompatible pooling or attention size')
        self.cnn = nn.Sequential(*layers)
        self.gru = nn.GRU(channels * freq, cfg.gru_hidden, cfg.gru_layers,
            batch_first=True, bidirectional=True, dropout=cfg.dropout if cfg.gru_layers > 1 else 0.)
        # Directional product is the DCASE CRNN convention; see reproduction notes.
        self.attention = nn.ModuleList([nn.MultiheadAttention(cfg.gru_hidden, cfg.attention_heads,
            dropout=cfg.dropout, batch_first=True) for _ in range(cfg.attention_layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(cfg.gru_hidden) for _ in range(cfg.attention_layers)])
        def head(size):
            return nn.Sequential(nn.Linear(cfg.gru_hidden, cfg.head_hidden), nn.Linear(cfg.head_hidden, size))
        if cfg.output_mode == 'multi_accddoa':
            self.output = head(cfg.tracks * cfg.classes * 4)
        else:
            self.doa_output = head(cfg.classes * 3)
            self.distance_output = head(cfg.classes)
        self.self_output = head(cfg.self_classes) if cfg.self_classes else None

    def forward(self, features):
        if features.ndim != 4 or features.shape[1] != self.cfg.input_channels or features.shape[-1] != self.cfg.frequency_bins:
            raise ValueError('Expected B x 4 x T x 512 binaural features')
        x = self.cnn(features).transpose(1, 2).contiguous().flatten(2)
        x, _ = self.gru(x)
        x = torch.tanh(x)
        h = self.cfg.gru_hidden
        x = x[..., :h] * x[..., h:]
        for attention, norm in zip(self.attention, self.norms):
            attended, _ = attention(x, x, x, need_weights=False)
            x = norm(x + attended)
        b, t, _ = x.shape
        if self.cfg.output_mode == 'multi_accddoa':
            result = {'accddoa': self.output(x).reshape(b, t, self.cfg.tracks, self.cfg.classes, 4)}
        else:
            result = {'accdoa': torch.tanh(self.doa_output(x)).reshape(b, t, self.cfg.classes, 3),
                      'distance': torch.relu(self.distance_output(x))}
        if self.self_output is not None:
            result['self_logits'] = self.self_output(x)
        return result
