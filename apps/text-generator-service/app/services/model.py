import keras


def load_model():
    return keras.models.load_model('models/mini-gpt')


def predict(model: keras.Model, value: str):
    return model.predict(value)
