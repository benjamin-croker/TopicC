from topicc import load_topicc


def main():
    model = load_topicc('output/example.topicc')
    while True:
        seq = input('Enter sequence\n> ')
        preds = model.predict(seq)
        print(f'Predictions: {preds}')


if __name__ == '__main__':
    main()
