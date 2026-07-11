"""Model definition for the XLM-RoBERTa-CNN (and BETO-CNN) hate-speech classifier.

The idea is to use a pre-trained multilingual transformer as a contextual
feature extractor and a small 2-D CNN as the classification head. Instead of
using only the last hidden state, the *thirteen* hidden states of the
transformer (the embedding layer plus the twelve encoder layers) are stacked
along the channel dimension and fed to the convolution, so the head can learn
which layers are most informative for the task.
"""

import torch
import torch.nn as nn
from transformers import BertModel, XLMRobertaModel

# A base transformer exposes 13 hidden states: the embeddings + 12 encoder layers.
NUM_HIDDEN_STATES = 13
HIDDEN_SIZE = 768

# Flattened size of the CNN head, tied to the tokenizer's MAX_LENGTH (300).
# conv (kernel (2, 768), padding 2) -> (13, 303, 5); max-pool (2, stride 1) -> (13, 302, 4)
# 13 * 302 * 4 = 15704. Change this if MAX_LENGTH changes.
FLATTENED_SIZE = 15704

# Hugging Face id for BETO (Spanish BERT).
BETO_ID = "dccuchile/bert-base-spanish-wwm-cased"


class TransformerCNN(nn.Module):
    """A transformer backbone with a 2-D CNN classification head.

    Args:
        backbone: ``"xlm-roberta-base"`` for the multilingual model, or
            ``"beto"`` for the Spanish-specific BERT baseline.
    """

    def __init__(self, backbone: str = "xlm-roberta-base"):
        super().__init__()

        if backbone == "beto":
            self.encoder = BertModel.from_pretrained(BETO_ID)
        else:
            self.encoder = XLMRobertaModel.from_pretrained(backbone)

        self.conv = nn.Conv2d(
            in_channels=NUM_HIDDEN_STATES,
            out_channels=NUM_HIDDEN_STATES,
            kernel_size=(2, HIDDEN_SIZE),
            padding=2,
        )
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1)
        self.dropout = nn.Dropout(0.1)
        self.flatten = nn.Flatten()
        self.classifier = nn.Linear(FLATTENED_SIZE, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        # Ask the transformer for every hidden state.
        _, _, hidden_states = self.encoder(
            input_ids,
            attention_mask,
            output_hidden_states=True,
            return_dict=False,
        )

        # Stack the 13 hidden states as channels: (batch, 13, seq_len, hidden).
        x = torch.stack(hidden_states, dim=1)

        # CNN head.
        x = self.dropout(x)
        x = self.pool(self.dropout(self.relu(self.conv(x))))
        x = self.dropout(self.flatten(x))
        x = self.classifier(x)

        # Single logit -> probability of the "toxic" class.
        return self.sigmoid(x)
