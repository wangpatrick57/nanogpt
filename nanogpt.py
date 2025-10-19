from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import wget


class DataLoader:
    def __init__(self, data: torch.Tensor, batch_size: int, context_length: int):
        self.data = data
        self.batch_size = batch_size
        self.context_length = context_length

    def get_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        batch_idxs = torch.randint(
            len(self.data) - self.context_length, (self.batch_size,)
        )
        X = torch.stack(
            [
                self.data[batch_idx : batch_idx + self.context_length]
                for batch_idx in batch_idxs
            ]
        )
        Y = torch.stack(
            [
                self.data[batch_idx + 1 : batch_idx + self.context_length + 1]
                for batch_idx in batch_idxs
            ]
        )
        return X, Y


class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size: int):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, vocab_size)

    def forward(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        logits = self.embed(X)
        loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), Y.view(-1))
        return loss


if __name__ == "__main__":
    # Constants and hyperparameters.
    torch.manual_seed(1337)
    data_url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    data_path = Path("tinyshakespeare.txt")
    train_ratio = 0.9
    batch_size = 4
    context_length = 8

    # Preprocess data.
    if not data_path.exists():
        wget.download(data_url, out=str(data_path))

    raw_text = open(data_path).read()
    ctoi = {c: i for i, c in enumerate(sorted(list(set(raw_text))))}
    itoc = {i: c for c, i in ctoi.items()}
    vocab_size = len(ctoi)
    data = torch.tensor([ctoi[c] for c in raw_text], dtype=torch.long)
    split_idx = int(len(data) * train_ratio)
    train_data = data[:split_idx]
    val_data = data[split_idx:]
    data_loader = DataLoader(train_data, batch_size, context_length)

    # Train model.
    model = BigramLanguageModel(vocab_size)
    X_batch, Y_batch = data_loader.get_batch()
    loss = model(X_batch, Y_batch)
    print(loss)
