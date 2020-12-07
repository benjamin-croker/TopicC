import sys
import json

from topicc.interface import train_topicc

USAGE_MESSAGE = "Usage: python -m topicc.train <param_file>"

def main():
    if len(sys.argv) != 2:
        print(USAGE_MESSAGE)
        sys.exit(1)
    
    with open(sys.argv[1]) as f:
        params = json.load(f)
    
    train_topicc(params)


if __name__ == '__main__':
    main()