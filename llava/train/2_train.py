import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from llava.train.train_run_2 import train

if __name__ == "__main__":
    train(attn_implementation=None)
