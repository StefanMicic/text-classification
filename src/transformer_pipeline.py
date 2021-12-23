import argparse

from tensorflow import keras

from utils import create_model, prepare_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vocab_size", type=int, default=20000)
    parser.add_argument("-m", "--max_len", type=int, default=20)
    args = parser.parse_args()
    (x_train, y_train), (x_val, y_val) = prepare_data(args.vocab_size, args.max_len)
    embed_dim = 32
    num_heads = 2
    ff_dim = 32
    try:
        model = keras.models.load_model("transformer_classification")
    except IOError:
        model = create_model(args.max_len, args.vocab_size, 'transformer', embed_dim, num_heads, ff_dim)
        model.summary()
        model.fit(x_train, y_train, batch_size=32, epochs=2)
        model.save("transformer_classification")

    model.evaluate(x_val, y_val)


if __name__ == '__main__':
    main()
