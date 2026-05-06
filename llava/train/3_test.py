import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from llava.train.test_run import test

if __name__ == "__main__":
    test(attn_implementation=None)
