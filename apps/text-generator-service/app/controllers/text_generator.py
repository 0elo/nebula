import keras

from app.services import model


def process_prompt(text_generator: keras.Model, prompt: str) -> str:
    return model.predict(text_generator, prompt)
