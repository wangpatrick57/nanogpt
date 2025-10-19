from pathlib import Path

import wget


def set_up_train_data() -> None:
    TRAIN_DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    TRAIN_DATA_PATH = Path("input.txt")

    if not TRAIN_DATA_PATH.exists():
        wget.download(TRAIN_DATA_URL)


if __name__ == "__main__":
    set_up_train_data()
