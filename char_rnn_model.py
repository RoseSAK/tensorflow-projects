#!/usr/bin/env python
# coding=utf-8

import tensorflow as tf

tf.enable_eager_execution()

import numpy as np
import os
import time


if tf.test.is_gpu_available():
    rnn = tf.keras.layers.CuDNNGRU
else:
    import functools

    rnn = functools.partial(
        tf.keras.layers.GRU, recurrent_activation='sigmoid')


# create the input and target text for a sequence e.g. from 'H e l l o' create 'H e l l' (input) and 'e l l o' (target)
def split_input_target(chunk):
    input_text = chunk[:-1]
    target_text = chunk[1:]
    return input_text, target_text


def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim,
                                  batch_input_shape=[batch_size, None]),
        rnn(rnn_units,
            return_sequences=True,
            recurrent_initializer='glorot_uniform',
            stateful=True),
        tf.keras.layers.Dense(vocab_size)
    ])
    return model


def preprocess_data(text):
    print("Length of text: %d characters" % len(text))
    vocab = sorted(set(text))
    print("%d unique characters" % len(vocab))
    char2idx = {u:i for i, u in enumerate(vocab)}
    idx2char = np.array(vocab)
    text_as_int = np.array([char2idx[c] for c in text])
    # print('{')
    # for char,_ in zip(char2idx, range(20)):
    #     print('  {:4s}: {:3d},'.format(repr(char), char2idx[char]))
    # print('  ...\n}')
    # print ('{} ---- characters mapped to int ---- > {}'.format(repr(text[:13]), text_as_int[:13]))

    # Prepare data for training
    seq_length = 100  # the maximum length sequence we want for a single input
    examples_per_epoch = len(text) // seq_length

    # Create training examples / targets
    char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)
    # I think we use this method for two reasons: 1) we need the data to be a tensorflow dataset object
    # then we can call all kinds of other tensorflow methods on it
    # 2) this creates a stream, which is better than creating a constant tensor

    # for i in char_dataset.take(5):  # the 'take' method is just the method of accessing elements in the dataset object
    #     print(idx2char[i])

    sequences = char_dataset.batch(seq_length+1, drop_remainder=True)

    dataset = sequences.map(split_input_target)  # this is a tensorflow implementation of map
    # for input_example, target_example in dataset.take(1):
    #     print ('Input data: ', repr(''.join(idx2char[input_example.numpy()])))
    #     print ('Target data:', repr(''.join(idx2char[target_example.numpy()])))

    # Create training batches
    BATCH_SIZE = 64
    steps_per_epoch = examples_per_epoch // BATCH_SIZE

    # Buffer size to shuffle the dataset
    # (TF data is designed to work with possibly infinite sequences,
    # so it doesn't attempt to shuffle the entire sequence in memory. Instead,
    # it maintains a buffer in which it shuffles elements).
    BUFFER_SIZE = 10000

    dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)

    # Build the model
    vocab_size = len(vocab)
    embedding_dim = 256
    rnn_units = 1024

    model = build_model(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        rnn_units=rnn_units,
        batch_size=BATCH_SIZE
    )

    for input_example_batch, target_example_batch in dataset.take(1):
        example_batch_predictions = model(input_example_batch)
        print(example_batch_predictions.shape, "# (batch_size, sequence_length, vocab_size)")
        example_batch_loss = tf.losses.sparse_softmax_cross_entropy(target_example_batch, example_batch_predictions)
        print("Prediction shape: ", example_batch_predictions.shape, " # (batch_size, sequence_length, vocab_size)")
        print("scalar_loss:      ", example_batch_loss.numpy())

    #model.summary()

    model.compile(
        optimizer=tf.train.AdamOptimizer(),
        loss=tf.losses.sparse_softmax_cross_entropy
    )

    # Directory where the checkpoints will be saved
    checkpoint_dir = './training_checkpoints'
    # Name of the checkpoint files
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_prefix,
        save_weights_only=True
    )

    EPOCHS=3

    history = model.fit(dataset.repeat(), epochs=EPOCHS, steps_per_epoch=steps_per_epoch,
                        callbacks=[checkpoint_callback])


def main():
    path_to_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "shakespeare.txt")
    text_data = open(path_to_file).read()
    preprocessed = preprocess_data(text_data)


if __name__ == "__main__":
    main()
