from typing import Any, Tuple

from keras.layers import Bidirectional, GlobalMaxPool1D, LSTM
from loguru import logger as log
from numpy import ndarray
from tensorflow import keras
from tensorflow.keras import layers

from models.positional_embedding import TokenAndPositionEmbedding
from models.transformer_block import TransformerBlock


def create_model(max_len: int,
                 vocab_size: int,
                 key_word: str = "transformer",
                 embed_dim: int = 32,
                 num_heads: int = 2,
                 ff_dim: int = 32) -> keras.Model:
    inputs = layers.Input(shape=(max_len,))
    embedding_layer = TokenAndPositionEmbedding(max_len, vocab_size, embed_dim)
    x = embedding_layer(inputs)

    if key_word == "transformer":
        transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)
        x = transformer_block(x)
        x = layers.GlobalAveragePooling1D()(x)
        x = layers.Dropout(0.1)(x)
        x = layers.Dense(20, activation="relu")(x)
    else:
        x = Bidirectional(LSTM(5, return_sequences=True))(x)
        x = GlobalMaxPool1D()(x)

    outputs = layers.Dense(2, activation="softmax")(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def prepare_data(vocab_size: int = 20000, max_len: int = 200) -> Tuple[Tuple[ndarray, Any], Tuple[ndarray, Any]]:
    (x_train, y_train), (x_val, y_val) = keras.datasets.imdb.load_data(num_words=vocab_size)
    log.info(f"{len(x_train)} Training sequences and {len(x_val)} Validation sequences")
    x_train = keras.preprocessing.sequence.pad_sequences(x_train, maxlen=max_len)
    x_val = keras.preprocessing.sequence.pad_sequences(x_val, maxlen=max_len)
    return (x_train, y_train), (x_val, y_val)
