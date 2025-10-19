from pathlib import Path

import wget


if __name__ == "__main__":
    TRAIN_DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    TRAIN_DATA_PATH = Path("input.txt")

    if not TRAIN_DATA_PATH.exists():
        wget.download(TRAIN_DATA_URL)
    raw_train_data_text = open(TRAIN_DATA_PATH).read()
    enc = {c: i for i, c in enumerate(sorted(list(set(raw_train_data_text))))}
    dec = {i: c for c, i in enc.items()}
    vocab_size = len(enc)
