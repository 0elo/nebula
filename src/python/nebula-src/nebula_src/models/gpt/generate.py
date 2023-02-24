import keras
import numpy as np
import tensorflow as tf


class TextGenerator(keras.callbacks.Callback):
    """A callback to generate text from a trained model.
    1. Feed some starting prompt to the model
    2. Predict probabilities for the next token
    3. Sample the next token and add it to the next input

    Arguments:
        max_tokens: Integer, the number of tokens to be generated after prompt.
        start_tokens: List of integers, the token indices for the starting prompt.
        index_to_word: List of strings, obtained from the TextVectorization layer.
        top_k: Integer, sample from the `top_k` token predictions.
        print_every: Integer, print after this many epochs.
    """

    def __init__(self, max_tokens, max_seq_size, start_tokens, index_to_word, top_k=10, print_every=1):
        self.max_tokens = max_tokens
        self.max_seq_size = max_seq_size
        self.start_tokens = start_tokens
        self.index_to_word = index_to_word
        self.print_every = print_every
        self.k = top_k

    def sample_from(self, logits: list[int]):
        logits, indices = tf.math.top_k(logits, k=self.k, sorted=True)
        indices = np.asarray(indices).astype("int32")
        preds = keras.activations.softmax(tf.expand_dims(logits, 0))[0]
        preds = np.asarray(preds).astype("float32")
        return np.random.choice(indices, p=preds)

    def detokenize(self, number: int):
        return self.index_to_word[number]

    def on_epoch_end(self, epoch: int, logs=None):
        start_tokens = [_ for _ in self.start_tokens]
        if (epoch + 1) % self.print_every != 0:
            return
        num_tokens_generated = 0
        tokens_generated = []
        while num_tokens_generated <= self.max_tokens:
            pad_len = self.max_seq_size - len(start_tokens)
            sample_index = len(start_tokens) - 1
            if pad_len < 0:
                x = start_tokens[: self.max_seq_size]
                sample_index = self.max_seq_size - 1
            elif pad_len > 0:
                x = start_tokens + [0] * pad_len
            else:
                x = start_tokens
            x = np.array([x])
            y, _ = self.model.predict(x)
            sample_token = self.sample_from(y[0][sample_index])
            tokens_generated.append(sample_token)
            start_tokens.append(sample_token)
            num_tokens_generated = len(tokens_generated)
        txt = " ".join([self.detokenize(_) for _ in self.start_tokens + tokens_generated])
        print(f"generated text:\n{txt}\n")


# Tokenize starting prompt
def generate_word_to_index_map(vocab: list[str]) -> dict:
    word_to_index = {}
    for index, word in enumerate(vocab):
        word_to_index[word] = index
    return word_to_index


def generate_text_gen_callback(max_seq_size: int, prompt: str, vocab: list[str]) -> TextGenerator:
    # start_prompt = "this movie is"
    word_to_index = generate_word_to_index_map(vocab=vocab)
    start_tokens = [word_to_index.get(_, 1) for _ in prompt.split()]
    num_tokens_generated = 40
    text_gen = TextGenerator(
        max_tokens=num_tokens_generated, max_seq_size=max_seq_size, start_tokens=start_tokens, index_to_word=vocab
    )
    return text_gen
