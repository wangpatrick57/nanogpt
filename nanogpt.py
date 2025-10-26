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
        x = torch.stack(
            [
                data[batch_idx : batch_idx + self.context_length]
                for batch_idx in batch_idxs
            ]
        )
        y = torch.stack(
            [
                data[batch_idx + 1 : batch_idx + self.context_length + 1]
                for batch_idx in batch_idxs
            ]
        )
        return x, y


class AttentionHead(nn.Module):
    def __init__(self, context_length: int, embed_dim: int, head_dim: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.query = nn.Linear(embed_dim, head_dim, bias=False)
        self.key = nn.Linear(embed_dim, head_dim, bias=False)
        self.value = nn.Linear(embed_dim, head_dim, bias=False)
        # TODO: figure out whether this needs to be a buffer
        self.tril = torch.tril(torch.ones(context_length, context_length))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        (batch_size, context_size, embed_dim) -> (batch_size, context_size, head_dim)
        """
        context_size = x.shape[-2]
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        logits = q @ k.transpose(-1, -2) / self.embed_dim**0.5
        masked_logits = logits.masked_fill(
            self.tril[:context_size, :context_size] == 0, float("-inf")
        )
        probs = F.softmax(masked_logits, dim=-1)
        out = probs @ v
        return out


class MultiHeadAttention(nn.Module):
    def __init__(
        self, context_length: int, embed_dim: int, num_heads: int, head_dim: int
    ):
        super().__init__()
        self.heads = nn.ModuleList(
            [
                AttentionHead(context_length, embed_dim, head_dim)
                for _ in range(num_heads)
            ]
        )
        self.project = nn.Linear(num_heads * head_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        (batch_size, context_size, embed_dim) -> (batch_size, context_size, embed_dim)
        """
        out = torch.cat([head(x) for head in self.heads], dim=-1)
        proj = self.project(out)
        return proj


class FeedForward(nn.Module):
    def __init__(self, embed_dim: int, hidden_dim: int):
        super().__init__()
        self.linear = nn.Linear(embed_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.project = nn.Linear(hidden_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        (batch_size, context_size, embed_dim) -> (batch_size, context_size, embed_dim)
        """
        preact = self.linear(x)
        act = self.relu(preact)
        proj = self.project(act)
        return proj


class TransformerBlock(nn.Module):
    def __init__(
        self,
        context_length: int,
        embed_dim: int,
        num_heads: int,
        head_dim: int,
        hidden_dim: int,
    ):
        super().__init__()
        self.attn = MultiHeadAttention(context_length, embed_dim, num_heads, head_dim)
        self.ffwd = FeedForward(embed_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        (batch_size, context_size, embed_dim) -> (batch_size, context_size, embed_dim)
        """
        attn_out = x + self.attn(x)
        out = x + self.ffwd(attn_out)
        return out


class TransformerLanguageModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        embed_dim: int,
        num_heads: int,
        head_dim: int,
        hidden_dim: int,
        num_blocks: int,
    ):
        super().__init__()
        self.context_length = context_length
        self.token_emb = nn.Embedding(vocab_size, embed_dim)
        self.pos_emb = nn.Embedding(context_length, embed_dim)
        self.blocks = nn.Sequential(
            *[
                TransformerBlock(
                    context_length, embed_dim, num_heads, head_dim, hidden_dim
                )
                for _ in range(num_blocks)
            ]
        )
        self.lm_head = nn.Linear(embed_dim, vocab_size)

    def forward(
        self, x: torch.Tensor, y: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        x is (batch_size, context_size, vocab_size)
        y is (batch_size, context_size)

        Returns logits, loss
        logits is (batch_size, context_size, vocab_size)
        loss is (1,)
        """
        token_embeds = self.token_emb(x)
        pos_embeds = self.pos_emb(torch.arange(x.shape[1]))
        embeds = token_embeds + pos_embeds
        out = self.blocks(embeds)
        logits = self.lm_head(out)
        loss = (
            None
            if y is None
            else F.cross_entropy(logits.view(-1, logits.shape[-1]), y.view(-1))
        )
        return logits, loss

    def generate(self, texts: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        for _ in range(max_new_tokens):
            contexts = texts[:, -self.context_length :]
            logits, _ = self(contexts)
            new_logits = logits[:, -1, :]
            probs = F.softmax(new_logits, dim=-1)
            tokens = torch.multinomial(probs, 1)
            texts = torch.cat((texts, tokens), dim=-1)
        return texts


@torch.no_grad()
def estimate_losses(
    model: TransformerLanguageModel, data_loader: DataLoader
) -> dict[DataSplit, float]:
    losses = dict()
    model.eval()
    for split in DataSplit:
        x_batch, y_batch = data_loader.get_batch(split)
        _, loss = model(x_batch, y_batch)
        losses[split] = loss.item()
    model.train()
    return losses


def main():
    # Misc. constants.
    torch.manual_seed(1337)
    data_url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    data_path = Path("tinyshakespeare.txt")

    # Training hyperparameters.
    train_ratio = 0.9
    batch_size = 32
    max_iters = 5000
    eval_interval = 500
    lr = 1e-3

    # Model hyperparameters.
    context_length = 8
    embed_dim = 32
    num_heads = 4
    head_dim = embed_dim // num_heads
    hidden_dim = embed_dim * 4
    num_blocks = 3

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
    model = TransformerLanguageModel(
        vocab_size,
        context_length,
        embed_dim,
        num_heads,
        head_dim,
        hidden_dim,
        num_blocks,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    est_loss_iters = []
    losses_over_time = defaultdict(list)

    for iter in tqdm(range(max_iters)):
        if iter % eval_interval == 0:
            est_loss_iters.append(iter)
            losses = estimate_losses(model, data_loader)
            for split, loss in losses.items():
                losses_over_time[split].append(loss)

        x_batch, y_batch = data_loader.get_batch(DataSplit.TRAIN)
        _, loss = model(x_batch, y_batch)
        optimizer.zero_grad()
        assert isinstance(loss, torch.Tensor)
        loss.backward()
        optimizer.step()
    print(f"{losses=}")

    for split, losses in losses_over_time.items():
        plt.plot(est_loss_iters, losses, label=split.value)
    plt.legend()
    plt.show()

    # Generate from model.
    contexts = torch.zeros(1, 1, dtype=torch.long)
    contexts = model.generate(contexts, 200)
    assert len(contexts) == 1
    context = contexts[0]
    print("".join([dec[tok] for tok in context.tolist()]))


if __name__ == "__main__":
    main()
