from attention_text_classificaiton.positional_embedding import TokenAndPositionEmbedding
from keras.layers import Bidirectional, GlobalMaxPool1D, LSTM
from loguru import logger as log
from tensorflow import keras
from tensorflow.keras import layers


def create_model(max_len, vocab_size, embed_dim):
    inputs = layers.Input(shape=(max_len,))
    embedding_layer = TokenAndPositionEmbedding(max_len, vocab_size, embed_dim)
    x = embedding_layer(inputs)
    x = Bidirectional(LSTM(5, return_sequences=True))(x)
    x = GlobalMaxPool1D()(x)
    outputs = layers.Dense(2, activation="softmax")(x)

    model = keras.Model(inputs=inputs, outputs=outputs)

    model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def main():
    vocab_size = 20000
    max_len = 200
    (x_train, y_train), (x_val, y_val) = keras.datasets.imdb.load_data(num_words=vocab_size)
    log.info(f"{len(x_train)} Training sequences and {len(x_val)} Validation sequences")
    x_train = keras.preprocessing.sequence.pad_sequences(x_train, maxlen=max_len)
    x_val = keras.preprocessing.sequence.pad_sequences(x_val, maxlen=max_len)

    embed_dim = 32
    try:
        model = keras.models.load_model("rnn_classification")
    except IOError:
        model = create_model(max_len, vocab_size, embed_dim)
        model.summary()
        model.fit(
            x_train, y_train, batch_size=32, epochs=2, validation_data=(x_val, y_val)
        )
        model.save("rnn_classification")
    # 782/782 [==============================] - 15s 19ms/step - loss: 0.3122 - accuracy: 0.8699


if __name__ == '__main__':
    main()
