import keras


def save_model(model: keras.Model, dir: str):
    model.save(dir)


def load_model(model: keras.Model, dir: str):
    return keras.models.load_model(dir)