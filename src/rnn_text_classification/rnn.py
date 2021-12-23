from keras.layers import (Bidirectional, Dense, GlobalMaxPool1D, Input, LSTM)
from keras.models import load_model, Model

try:
    model = load_model("cnn_model")
    print("CNN MODEL LOADED")
except Exception as e:
    print(e)

    print("Building model...")

    input_ = Input(shape=(MAX_SEQUENCE_LENGTH,))
    x = embedding_layer(input_)
    x = Bidirectional(LSTM(15, return_sequences=True))(x)
    x = GlobalMaxPool1D()(x)
    output = Dense(len(possible_labels), activation="softmax")(x)
    model = Model(input_, output)
    model.compile(
        loss="categorical_crossentropy",
        optimizer="adam",
        metrics=["accuracy"],
    )

    print("Training model...")
    r = model.fit(
        data,
        targets,
        batch_size=BATCH_SIZE,
        epochs=20,
        validation_split=VALIDATION_SPLIT,
    )
    model.save("cnn_model")
