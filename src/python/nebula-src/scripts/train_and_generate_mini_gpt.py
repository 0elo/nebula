import keras.models

from nebula_src.models import gpt

vocab_size = 20000  # Only consider the top 20k words
max_seq_size = 80  # Max sequence size
embed_dim = 256  # Embedding size for each token
num_heads = 2  # Number of attention heads
feed_forward_dim = 256  # Hidden layer size in feed forward network inside transformer
batch_size = 128


def train_and_generate():
    model = gpt.model.create_model(
        embed_dim=embed_dim,
        feed_forward_dim=feed_forward_dim,
        max_seq_size=max_seq_size,
        num_heads=num_heads,
        vocab_size=vocab_size
    )
    print('Model created.')
    ds, vocab = gpt.train.prepare_dataset(batch_size=batch_size, max_seq_size=max_seq_size, vocab_size=vocab_size)
    print('Dataset prepared.')
    text_gen = gpt.generate.generate_text_gen_callback(max_seq_size=max_seq_size, prompt='hello world', vocab=vocab)
    print('Text generator callback created.')
    model.fit(ds, verbose=2, epochs=25, callbacks=[text_gen])

    model.save('nebula_models/gpt')


def load():
    return keras.models.load_model('nebula_models/gpt')


def predict(model: keras.Model, prompt: str) -> str:
    return model.predict(prompt)


if __name__ == '__main__':
    train_and_generate()
