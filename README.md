# RNN with Keras: Sentiment Analysis and NER, Code Explained

One guide for two notebooks: `rnn_sentiment_keras.ipynb` and `ner_keras.ipynb`. Both use the same idea (an RNN that reads words in order and keeps memory) but give different kinds of answers.

## Contents

1. [Part 1: Sentiment analysis](#part-1-sentiment-analysis)
2. [Part 2: Named entity recognition (NER)](#part-2-named-entity-recognition-ner)
3. [Sentiment vs NER at a glance](#sentiment-vs-ner-at-a-glance)

---

## Part 1: Sentiment analysis

Companion guide for `rnn_sentiment_keras.ipynb`. It explains what each part of the code does and why.

### What the notebook does

It reads a short sentence word by word and predicts whether it is **positive (1)** or **negative (0)**.

- Data: 50 labelled sentences (25 positive, 25 negative), including "i like her attitude"
- Model: Embedding, SimpleRNN, Dense (sigmoid)
- Output: one answer per sentence

```
i  like  her  attitude   ->  [RNN reads each word, keeps memory]  ->  positive (0.97)
```

### How to run

```bash
pip install tensorflow matplotlib
jupyter notebook rnn_sentiment_keras.ipynb
```

Run the cells from top to bottom.

### Code walkthrough

#### 1. Imports and seed

```python
keras.utils.set_random_seed(42)
```

Neural networks start with random weights. Fixing the seed makes the run repeatable, so you get the same split and similar results each time.

#### 2. The dataset

Two Python lists hold the positive and negative sentences, and `labels` marks them 1 or 0.

**Why:** a model learns from examples that have the right answer attached. I reused the same sentiment words (love, hate, terrible, wonderful) across several sentences so that a model this small has a chance to learn them. With a one-off word, it sees only a single example and cannot generalise.

#### 3. Train/test split

```python
idx = np.random.permutation(len(texts))
test_idx, train_idx = idx[:10], idx[10:]
```

40 sentences are used to train and 10 are hidden. **Why:** if you only check the model on sentences it already studied, you cannot tell whether it learned the pattern or just memorised the answers.

#### 4. `TextVectorization`

```python
vectorizer = layers.TextVectorization(output_mode="int", output_sequence_length=MAXLEN)
vectorizer.adapt(tf.constant(train_texts))
```

- Neural networks only understand numbers, so each word becomes an ID.
- `output_sequence_length=MAXLEN` pads short sentences with zeros so all inputs have the same length.
- `adapt` builds the vocabulary from the **training** sentences only. Test words the model never saw become `[UNK]` (id 1). This copies what happens on real, unseen data.

#### 5. The model

```python
model = keras.Sequential([
    keras.Input(shape=(MAXLEN,)),
    layers.Embedding(VOCAB, 16, mask_zero=True),
    layers.SimpleRNN(24, activation="tanh"),
    layers.Dense(1, activation="sigmoid"),
])
```

| Layer | What it does | Why |
|---|---|---|
| `Embedding` | Turns each word ID into 16 learnable numbers | Words with similar sentiment can end up with similar numbers |
| `mask_zero=True` | Ignores padding zeros | The model should not learn from empty filler |
| `SimpleRNN(24)` | Reads words in order, keeps a 24-number memory, returns the memory after the last word | This is the "memory" idea from the slides: hₜ = tanh(W·xₜ + U·hₜ₋₁ + b) |
| `Dense(1, sigmoid)` | Squashes the final memory into a number between 0 and 1 | Close to 1 means positive, close to 0 means negative |

#### 6. Compile

```python
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.01, clipnorm=5.0),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)
```

- `binary_crossentropy`: the loss for a yes/no answer. It gives a big penalty for being confidently wrong.
- `Adam`: the optimiser that adjusts the weights after each batch.
- `clipnorm=5.0`: caps the size of any single update. RNN gradients can explode over many steps, so this is a safety limit.

#### 7. Training

```python
history = model.fit(X_train, y_train, validation_data=(X_test, y_test),
                    epochs=60, batch_size=8, verbose=0)
```

One call does the whole loop from the training slides: forward pass, loss, backpropagation through time and weight update. It repeats this for 60 epochs (full passes over the data) in batches of 8 sentences. `history` records the loss and accuracy after every epoch.

#### 8. Learning curves

The two plots show loss going down and accuracy going up. **Why:** if the train line improves but the test line does not, the model is memorising. This is called overfitting.

#### 9. Test predictions

```python
probs = model.predict(X_test, verbose=0).ravel()
```

Each sentence gets a probability of being positive. Anything above 0.5 counts as positive. The loop prints OK or BAD for each test sentence.

#### 10. `predict()` for your own sentence

It turns your text into IDs with the same vectorizer, runs the model and returns "positive" or "negative" with the probability. Use words from the dataset, because unknown words carry no sentiment.

### Results and limits

- A typical run reaches about 100% training accuracy and around 70% test accuracy. With only 10 test sentences, this number is noisy and changes with the seed.
- Some sentiment words appear only once in the data, so the model cannot learn them from a single example.
- A plain RNN struggles with negation ("i do not like this movie") and with long sentences, because gradients fade over many steps.

### Ways to improve it

1. Use more data, for example the IMDB movie review dataset.
2. Swap `SimpleRNN` for `LSTM` or `GRU`.
3. Wrap it as `Bidirectional(LSTM(24))` to read the sentence in both directions.
4. Use pretrained word vectors instead of learning embeddings from 40 sentences.

---

## Part 2: Named entity recognition (NER)

Companion guide for `ner_keras.ipynb`. It explains what each part of the code does and why.

### What the notebook does

Sentiment analysis gives **one label per sentence**. NER gives **one label per word**: the model marks which words are people, organisations or locations.

```
Alice   works  at  Google  in  London
B-PER   O      O   B-ORG   O   B-LOC
```

- Data: 50 short annotated sentences with `PER`, `ORG` and `LOC` entities
- Model: Embedding, Bidirectional SimpleRNN (returns every step), Dense (softmax)
- Output: one tag for every word

#### The BIO tags

| Tag | Meaning |
|---|---|
| `B-PER` | **B**egins a person name |
| `I-PER` | **I**nside (continues) a person name, as in the second word of "John Smith" |
| `B-ORG`, `I-ORG` | Same idea for organisations |
| `B-LOC`, `I-LOC` | Same idea for locations, as in "New York" |
| `O` | **O**utside any entity |

The `B-` and `I-` split lets the model tell "New York" (one place) apart from two places next to each other.

### How to run

```bash
pip install tensorflow matplotlib
jupyter notebook ner_keras.ipynb
```

Run the cells from top to bottom.

### Code walkthrough

#### 1. Writing the dataset as `[PER Alice] works at [ORG Google]`

```python
"[PER Alice] works at [ORG Google] in [LOC London]"
```

**Why:** typing a tag for every word by hand is slow and easy to get wrong. Instead I mark the entities inside the sentence, and the `parse()` function converts them into word-level tags automatically:

```python
tokens: Alice  works at Google in London
tags:   B-PER  O     O  B-ORG  O  B-LOC
```

Multi-word names such as "John Smith" become `B-PER I-PER`.

#### 2. Train/test split

40 sentences train the model and 10 are hidden. **Why:** this tells you whether the model learned a pattern or just memorised the training sentences.

#### 3. Vectorizing and padding

```python
vectorizer = layers.TextVectorization(standardize=None, split="whitespace",
                                      output_mode="int", output_sequence_length=MAXLEN)
```

- Words become ID numbers because networks only understand numbers.
- `standardize=None` keeps **capital letters**. "Google" and "google" stay different, and a capital letter is a strong clue that a word is a name.
- Short sentences are padded with zeros to length `MAXLEN` so every input is the same size.
- The tags are padded the same way, so word 3 always lines up with tag 3.
- The vocabulary comes from the training split only.

#### 4. The model

```python
model = keras.Sequential([
    keras.Input(shape=(MAXLEN,)),
    layers.Embedding(VOCAB, 32, mask_zero=True),
    layers.Bidirectional(layers.SimpleRNN(32, return_sequences=True)),
    layers.Dense(len(tag_names), activation="softmax"),
])
```

| Layer | What it does | Why |
|---|---|---|
| `Embedding` | Turns each word ID into 32 learnable numbers | The model can learn that Paris and London behave alike |
| `mask_zero=True` | Ignores the padding zeros | Padding should not affect the loss |
| `SimpleRNN(..., return_sequences=True)` | Outputs a memory state at **every** word | NER needs a tag per word, so I keep every step, not only the last one |
| `Bidirectional` | Reads left-to-right and right-to-left | In "works at Google in London", the words before and after a word are both clues |
| `Dense(7, softmax)` | Gives a probability for each of the 7 tags at every word | The highest probability becomes that word's tag |

#### 5. Compile and train

```python
loss="sparse_categorical_crossentropy"
optimizer=keras.optimizers.Adam(learning_rate=0.01, clipnorm=5.0)
```

- `sparse_categorical_crossentropy`: the multi-class version of the loss used in the sentiment notebook. "Sparse" means the correct tag is given as a number (such as 3) instead of a list of zeros and ones.
- `clipnorm=5.0`: caps the size of any single weight update, which keeps RNN training stable.
- `model.fit(...)` runs the full loop for 60 epochs: forward pass, loss, backpropagation through time and weight update.

#### 6. Why entity-level scoring

```python
def get_entities(seq): ...
def entity_scores(true_tags, pred_tags): ...
```

**Why:** most words are `O`. A lazy model that tags everything `O` would still score high accuracy. The notebook computes precision, recall and F1 on **whole entities**: a prediction counts only if both the type and the full span match.

| Metric | Question it answers |
|---|---|
| Precision | Of the entities the model found, how many were correct? |
| Recall | Of the real entities, how many did the model find? |
| F1 | A single score that balances the two |

Example: if the truth is "New York" as `LOC` and the model tags only "New", that is a miss, even though one word was right.

#### 7. Test predictions and `tag_sentence()`

The prediction loop prints each test sentence with its tags and marks OK or BAD. For BAD cases it also prints the true tags. `tag_sentence()` lets you type your own sentence and shows the tags and the entities found:

```
Priya/B-PER works/O at/O Google/B-ORG in/O Paris/B-LOC
entities: [('PER', 'Priya'), ('ORG', 'Google'), ('LOC', 'Paris')]
```

### Results and limits

- The dataset is a toy, and 10 test sentences give a noisy score.
- The same names appear in many sentences, so the model partly memorises them.
- It cannot recognise a name it never saw. Unknown words become `[UNK]`, and a plain embedding cannot tell that "Delhi" is a place.

### Ways to improve it

1. Add features that generalise to new names: character-level embeddings, a capitalisation flag, or pretrained word vectors.
2. Use `LSTM` or `GRU` instead of `SimpleRNN`.
3. Add a CRF layer on top, which keeps tag sequences valid (for example, no `I-PER` straight after `O`).
4. Fine-tune a transformer such as BERT on a real dataset like CoNLL-2003.

---

## Sentiment vs NER at a glance


| | Sentiment | NER |
|---|---|---|
| Output | One label per sentence | One tag per word |
| RNN setting | Last state only | `return_sequences=True` |
| Direction | One way | `Bidirectional` |
| Last layer | `Dense(1, sigmoid)` | `Dense(7, softmax)` |
| Loss | `binary_crossentropy` | `sparse_categorical_crossentropy` |
