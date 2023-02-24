import keras


def load_model():
    return keras.models.load_model('../../src/python/nebula-src/nebula_models/gpt')


def predict(model: keras.Model, value: str):
    return model.predict(value)
