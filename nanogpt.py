from collections import defaultdict
from pathlib import Path
from enum import Enum

from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import wget


class DataSplit(Enum):
    TRAIN = "train"
    VALIDATION = "validation"


class DataLoader:
    def __init__(
        self,
        data: torch.Tensor,
        train_ratio: float,
        batch_size: int,
        context_length: int,
    ):
        split_idx = int(len(data) * train_ratio)
        self.data_map = {
            DataSplit.TRAIN: data[:split_idx],
            DataSplit.VALIDATION: data[split_idx:],
        }
        self.batch_size = batch_size
        self.context_length = context_length

    def get_batch(self, split: DataSplit) -> tuple[torch.Tensor, torch.Tensor]:
        data = self.data_map[split]
        batch_idxs = torch.randint(len(data) - self.context_length, (self.batch_size,))
        X = torch.stack(
            [
                data[batch_idx : batch_idx + self.context_length]
                for batch_idx in batch_idxs
            ]
        )
        Y = torch.stack(
            [
                data[batch_idx + 1 : batch_idx + self.context_length + 1]
                for batch_idx in batch_idxs
            ]
        )
        return X, Y


class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size: int):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, vocab_size)

    def forward(
        self, X: torch.Tensor, Y: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        logits = self.embed(X)
        loss = (
            None
            if Y is None
            else F.cross_entropy(logits.view(-1, logits.shape[-1]), Y.view(-1))
        )
        return logits, loss

    def generate(self, contexts: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        for _ in range(max_new_tokens):
            logits, _ = self(contexts)
            new_logits = logits[:, -1, :]
            probs = F.softmax(new_logits, dim=-1)
            tokens = torch.multinomial(probs, 1)
            contexts = torch.cat((contexts, tokens), dim=-1)
        return contexts


@torch.no_grad()
def estimate_losses(
    model: BigramLanguageModel, data_loader: DataLoader
) -> dict[DataSplit, float]:
    losses = dict()
    model.eval()
    for split in DataSplit:
        X_batch, Y_batch = data_loader.get_batch(split)
        _, loss = model(X_batch, Y_batch)
        losses[split] = loss.item()
    model.train()
    return losses


if __name__ == "__main__":
    # Misc. constants.
    torch.manual_seed(1337)
    data_url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    data_path = Path("tinyshakespeare.txt")

    # Training hyperparameters.
    train_ratio = 0.9
    batch_size = 32
    max_iters = 3000
    eval_interval = 300
    lr = 1e-2

    # Model hyperparameters.
    context_length = 8
    embed_dim = 32

    # Preprocess data.
    if not data_path.exists():
        wget.download(data_url, out=str(data_path))

    raw_text = open(data_path).read()
    enc = {ch: tok for tok, ch in enumerate(sorted(list(set(raw_text))))}
    dec = {tok: ch for ch, tok in enc.items()}
    vocab_size = len(enc)
    data = torch.tensor([enc[c] for c in raw_text], dtype=torch.long)
    data_loader = DataLoader(data, train_ratio, batch_size, context_length)

    # Train model.
    model = BigramLanguageModel(vocab_size)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    est_loss_iters = []
    losses_over_time = defaultdict(list)

    for iter in tqdm(range(max_iters)):
        if iter % eval_interval == 0:
            est_loss_iters.append(iter)
            losses = estimate_losses(model, data_loader)
            for split, loss in losses.items():
                losses_over_time[split].append(loss)

        X_batch, Y_batch = data_loader.get_batch(DataSplit.TRAIN)
        _, loss = model(X_batch, Y_batch)
        optimizer.zero_grad()
        assert isinstance(loss, torch.Tensor)
        loss.backward()
        optimizer.step()

    for split, losses in losses_over_time.items():
        plt.plot(est_loss_iters, losses, label=split.value)
    plt.legend()
    plt.show()

    # Generate from model.
    contexts = torch.zeros(1, 1, dtype=torch.long)
    contexts = model.generate(contexts, 100)
    assert len(contexts) == 1
    context = contexts[0]
    print("".join([dec[tok] for tok in context.tolist()]))
