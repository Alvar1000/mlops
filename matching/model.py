import torch
import torch.nn as nn


class PairwiseBinaryClassifier(nn.Module):
    def __init__(
        self, text_emb_size: int, img_emb_size: int, hidden_size: int, nlayers: int
    ) -> None:
        super(PairwiseBinaryClassifier, self).__init__()
        input_size = 2 * (text_emb_size + img_emb_size)
        layers = []
        for i in range(nlayers):
            layers.extend(
                [
                    nn.Linear(input_size if i == 0 else hidden_size, hidden_size),
                    nn.BatchNorm1d(hidden_size),
                    nn.PReLU(),
                ]
            )
        self.layers = nn.Sequential(*layers)
        self.scorer = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()
        self._init_params()

    def _init_params(self):
        for param in self.parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)

    def forward(self, text_emb1, img_emb1, text_emb2, img_emb2):
        x = torch.cat((text_emb1, img_emb1, text_emb2, img_emb2), dim=-1)
        x = self.layers(x)
        x = self.sigmoid(self.scorer(x))
        return x
