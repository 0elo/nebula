import tensorflow as tf
from keras import layers

import os
import string
import random
import typing


# batch_size = 128
def prepare_dataset(batch_size: int, max_seq_size: int, vocab_size: int) -> typing.Tuple[tf.data.TextLineDataset, list[str]]:
    # The dataset contains each review in a separate text file
    # The text files are present in four different folders
    # Create a list all files
    # filenames = []
    directories = [
        "datasets/aclImdb/train/pos",
        "datasets/aclImdb/train/neg",
        "datasets/aclImdb/test/pos",
        "datasets/aclImdb/test/neg",
    ]
    file_names = _get_files_from_dirs(directories=directories)
    ds = _create_dataset_from_text_files(batch_size=batch_size, file_names=file_names)
    vectorize_layer = _create_vectorization_layer(max_seq_size=max_seq_size, vocab_size=vocab_size)
    vectorize_layer.adapt(ds)
    vocab = vectorize_layer.get_vocabulary()  # To get words back from token indices
    ds = ds.map(_prepare_lm_inputs_labels(vectorize_layer=vectorize_layer))
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds, vocab


def _get_files_from_dirs(directories: list[str]) -> list[str]:
    file_names = []
    for d in directories:
        for f in os.listdir(d):
            file_names.append(os.path.join(d, f))
    return file_names


# Create a dataset from text files
def _create_dataset_from_text_files(batch_size: int, file_names: list[str]) -> tf.data.TextLineDataset:
    random.shuffle(file_names)
    text_ds = tf.data.TextLineDataset(file_names)
    text_ds = text_ds.shuffle(buffer_size=256)
    text_ds = text_ds.batch(batch_size)
    return text_ds


def _custom_standardization(input_string: str) -> str:
    """ Remove html line-break tags and handle punctuation """
    lowercased = tf.strings.lower(input_string)
    stripped_html = tf.strings.regex_replace(lowercased, "<br />", " ")
    return tf.strings.regex_replace(stripped_html, f"([{string.punctuation}])", r" \1")


# Create a vectorization layer and adapt it to the text
def _create_vectorization_layer(max_seq_size: int, vocab_size: int) -> layers.TextVectorization:
    return layers.TextVectorization(
        standardize=_custom_standardization,
        max_tokens=vocab_size - 1,
        output_mode="int",
        output_sequence_length=max_seq_size + 1,
    )


def _prepare_lm_inputs_labels(vectorize_layer: layers.TextVectorization) -> typing.Callable:
    """
    Shift word sequences by 1 position so that the target for position (i) is
    word at position (i+1). The model will use all words up till position (i)
    to predict the next word.
    """
    def _inner(text: str) -> typing.Tuple[str, str]:
        text = tf.expand_dims(text, -1)
        tokenized_sentences = vectorize_layer(text)
        x = tokenized_sentences[:, :-1]
        y = tokenized_sentences[:, 1:]
        return x, y
    return _inner





