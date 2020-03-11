# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
from headliner.trainer import Trainer
from headliner.preprocessing.preprocessor import Preprocessor
from headliner.model.transformer_summarizer import TransformerSummarizer
from tensorflow_datasets.core.features.text import SubwordTextEncoder

# constants
max_sequence_len = 512
embedding_size = 512
batch_size = 256
learning_rate = 1e-4
dropout_rate = 0.1
heads = 8
layers = 6
dim = 512

data = []
token_set = set()

with open("./dataset/local/integration.in", "r") as f:
    in_data = [line.strip() for line in f.readlines()]

with open("./dataset/local/integration.out", "r") as f:
    out_data = [line.strip() for line in f.readlines()]

assert len(in_data) == len(out_data)

for i in np.random.permutation(len(in_data)):
    data.append((in_data[i], out_data[i]))
    for token in in_data[i].split(',') + out_data[i].split(','):
        token_set.add(token)

print("Total tokens: ", len(token_set))
print("Total dataset: ", len(data))

# divide dataset
train_data = data[: int(len(data) * 0.8)]
valid_data = data[int(len(data) * 0.8) : int(len(data) * 0.9)]
test_data = data[int(len(data) * 0.9) :]

preprocessor = Preprocessor(filter_pattern='', punctuation_pattern='([,])', lower_case=True, hash_numbers=False)

"""
train_prep = [preprocessor(t) for t in train_data]
inputs_prep = [t[0] for t in train_prep]
targets_prep = [t[1] for t in train_prep]


tokenizer_input = SubwordTextEncoder.build_from_corpus(
    inputs_prep,
    target_vocab_size=len(token_set),
    reserved_tokens=[preprocessor.start_token, preprocessor.end_token]
)

tokenizer_target = SubwordTextEncoder.build_from_corpus(
    targets_prep,
    target_vocab_size=len(token_set),
    reserved_tokens=[preprocessor.start_token, preprocessor.end_token]
)

vectorizer = Vectorizer(tokenizer_input, tokenizer_target)
"""

summarizer = TransformerSummarizer(
    embedding_size=embedding_size,
    max_prediction_len=max_sequence_len,
    max_sequence_len=max_sequence_len,
    num_layers=layers,
    num_heads=heads,
    dropout_rate=dropout_rate,
    feed_forward_dim=dim
)
summarizer.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
#summarizer.optimizer_encoder = tf.keras.optimizers.Adam(learning_rate=learning_rate)
#summarizer.optimizer_decoder = tf.keras.optimizers.Adam(learning_rate=learning_rate)
#summarizer.init_model(preprocessor, vectorizer)

trainer = Trainer(
    batch_size=batch_size,
    max_input_len=max_sequence_len,
    max_output_len=max_sequence_len,
#    max_vovab_size_encoder=len(token_set),
#    max_vovab_size_decoder=len(token_set),
    steps_per_epoch=125,
    tensorboard_dir='/tmp/tensorboard',
    model_save_path='/tmp/summarizer'
)

trainer.train(summarizer, train_data, val_data=valid_data, num_epochs=150)

