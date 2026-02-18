import torch.nn as nn


class AudioToFlameMapper(nn.Module):
    def __init__(self, input_dim=1024, expression_dim=50, jaw_dim=3):
        super(AudioToFlameMapper, self).__init__()

        self.mapping = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, expression_dim + jaw_dim)
        )

    def forward(self, x):
        # x to cechy z HuBERTa [Batch, T, 1024]
        out = self.mapping(x)
        # out to parametry FLAME [Batch, T, 53]
        return out


