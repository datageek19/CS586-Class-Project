import csv
from nltk.tokenize import word_tokenize, sent_tokenize
from collections import Counter, defaultdict
from tqdm import tqdm
import string
import numpy as np

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

rng = np.random

# Parameters
learning_rate = 0.01
training_epochs = 20
display_step = 25

# Network Parameters
batch_size = 300
vocab_size = 10000
num_hiddenLay_1 = 256
num_hiddenLay_2 = 256
n_embedding = 256
positive_samples = 4

data_index = 0
window_size = 25
half_window = round((window_size - 1) / 2)
num_skips = 2
words_per_batch = round(batch_size / num_skips)

###
import nltk
nltk.download('punkt')

def get_words(file_loc):
    with open(file_loc, encoding='utf-8') as f:
        sentences = csv.reader(f)
        words=[]
        for sentence in sentences:
            tokenized_sents = [word_tokenize(i) for i in sentence]
            for token in tokenized_sents:
                for each in token:
                    lower_text = each.lower()
                    punc = string.punctuation
                    punc = punc.replace('-','')
                    punc = punc.replace('\'','')
                    punc = list(punc)
                    for p in punc:
                        lower_text = lower_text.replace(p,' ')
                    lower_text = lower_text.replace('-',' ')
                    words.append(lower_text)
        return words

## get words
train_words= get_words(file_loc='./data/train.pair_tok.tsv')
vali_words = get_words(file_loc='./data/dev.pair_tok.tsv')

### generate trigram words based on wordhashing method
def explode_word(word):
    w = "#" + word + "#"
    for i in range(0, len(w)-3+1):
        yield w[i:i+3]
##
def word2tri(words):
    count=Counter()
    for w in tqdm(words):
        count.update(explode_word(w))
    count = [["###", -1]] + count.most_common(vocab_size - 1)
    dictionary = defaultdict(lambda: 0)
    for key, val in enumerate(count):
        dictionary[val[0]] = key
    return dictionary.keys()

train_trigram = word2tri(train_words)
vali_trigram = word2tri(vali_words)

## generate batch
def generate_batch():
    global data_index
    buffer_x = [] # batch_size X vocab_size
    buffer_y = [] # batch_size X positive_samples
    for _ in range(words_per_batch):
        x = [0] * vocab_size
        words = train_words[data_index + half_window]
        trigram = word2tri(words)

        for i in range(len(trigram)):
            x[i] = 1
        for _ in range(num_skips):
            y = [0] * positive_samples
        
            sample_word = rng.randint(-half_window, half_window - 1)
            if half_window >= 0: 
                sample_word += 1
            sample_tris = list(word2tri(train_words[data_index + half_window + sample_word]))
            for i, k in enumerate(rng.choice(sample_tris, positive_samples)):
                y[i] = k
            buffer_x.append(x)
            buffer_y.append(y)
        data_index = (data_index + 1) % (vocab_size - window_size)
    return buffer_x, buffer_y
    
X, Y = generate_batch()

### dssm model setup
# Model configuration
layer_input = tf.placeholder("float", [batch_size, vocab_size], "layer_input")
layer_output_nums = tf.placeholder(tf.int32, [batch_size, positive_samples], "layer_output")

w_l1 = tf.Variable(tf.random_normal([vocab_size, num_hiddenLay_1]))
w_l1_bias = tf.Variable(tf.random_normal([num_hiddenLay_1]))
layer_1_sum = tf.add(tf.matmul(layer_input, w_l1), w_l1_bias)
layer_1 = tf.nn.relu(layer_1_sum)

w_l2 = tf.Variable(tf.random_normal([num_hiddenLay_1, num_hiddenLay_2]))
w_l2_bias = tf.Variable(tf.random_normal([num_hiddenLay_2]))
layer_2_sum = tf.add(tf.matmul(layer_1, w_l2), w_l2_bias)

w_out = tf.Variable(tf.random_normal([vocab_size, num_hiddenLay_2]))
w_out_bias = tf.Variable(tf.random_normal([vocab_size]))

loss = tf.nn.sampled_softmax_loss(w_out, w_out_bias, layer_output_nums, layer_2_sum, 
                                  num_sampled = batch_size, num_classes = vocab_size, num_true = positive_samples)
cost = tf.reduce_mean(loss)
optimizer = tf.train.AdamOptimizer(learning_rate=0.01).minimize(cost)

init = tf.global_variables_initializer()
train_inputs = tf.placeholder(tf.float64, shape=[batch_size, vocab_size], name = "train_inputs")

### run baseline model

np.set_printoptions(precision=3)

with tf.Session() as sess:
    sess.run(init)
    # Training cycle
    for epoch in range(training_epochs):
        cc = []
        for _ in range(round((vocab_size - window_size) / words_per_batch)):
            batch_x, batch_y = generate_batch()
            _, c = sess.run([optimizer, cost], feed_dict={layer_input: batch_x, layer_output_nums: batch_y})
            cc.append(c)
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(np.mean(np.array(cc))))