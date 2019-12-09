import json
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import numpy as np
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
from tqdm import tqdm


# Initialization
rng = np.random

# Parameters
learning_rate = 0.001
training_epochs = 1000
display_step = 5

# Network Parameters
batch_size = 4
tot_vocab = 10000
num_hiddenLay_1 = 256
num_hiddenLay_2 = 256
n_embedding = 256
positive_samples = 4

data_index = 0
window_size = 5
half_window = round((window_size - 1) / 2)
num_skips = 2
words_per_batch = round(batch_size / num_skips)


# Model itself
layer_input = tf.placeholder("float", [batch_size, tot_vocab], "layer_input")
layer_output_nums = tf.placeholder(tf.int32, [batch_size, positive_samples], "layer_output")

w_l1 = tf.Variable(tf.random_normal([tot_vocab, num_hiddenLay_1]))
w_l1_bias = tf.Variable(tf.random_normal([num_hiddenLay_1]))
layer_1_sum = tf.add(tf.matmul(layer_input, w_l1), w_l1_bias)
layer_1 = tf.nn.relu(layer_1_sum)


w_l2 = tf.Variable(tf.random_normal([num_hiddenLay_1, num_hiddenLay_2]))
w_l2_bias = tf.Variable(tf.random_normal([num_hiddenLay_2]))
layer_2_sum = tf.add(tf.matmul(layer_1, w_l2), w_l2_bias)

w_out = tf.Variable(tf.random_normal([tot_vocab, num_hiddenLay_2]))
w_out_bias = tf.Variable(tf.random_normal([tot_vocab]))

loss = tf.nn.sampled_softmax_loss(w_out, w_out_bias, layer_output_nums, layer_2_sum, 
                                  num_sampled = batch_size, num_classes = tot_vocab, num_true = positive_samples)
cost = tf.reduce_mean(loss)
optimizer = tf.train.AdamOptimizer(learning_rate=0.01).minimize(cost)

init = tf.global_variables_initializer()
train_inputs = tf.placeholder(tf.float64, shape=[batch_size, tot_vocab], name = "train_inputs")

## prepare data
data_index = 0

def explode_word(word):
    w = "#" + word + "#"
    for i in range(0, len(w)-3+1):
        yield w[i:i+3]


def load_vocab(file_path):
    word_dict = {}
    with open(file_path, encoding='utf8') as f:
        for idx, word in enumerate(f.readlines()):
            word = word.strip()
            word_dict[word] = idx
    return word_dict

## get words
vocab_outputView=load_vocab(file_path="./data/oppo_round1_train_20180929.txt")

## to see partial view of word dictionary
res= dict(list(vocab_outputView.items())[0:5])
print(res)

## get words
def gen_word_set(file_path):
    word_set = set()
    with open(file_path, encoding='utf8') as f:
        for line in f.readlines():
            spline = line.strip().split('\t')
            if len(spline) < 4:
                continue
            prefix, query_pred, title, tag, label = spline
            if label == '0':
                continue
            cur_arr = [prefix, title]
            query_pred = json.loads(query_pred)
            for w in prefix:
                word_set.add(w)
            for each in query_pred:
                for w in each:
                    word_set.add(w)
    return list(word_set)

##
file_vali = './data/oppo_round1_vali_20180929.txt'
chi_wordset= gen_word_set(file_path=file_vali)

def explode_word(word):
    w = "#" + word + "#"
    for i in range(0, len(w)-3+1):
        yield w[i:i+3]
##
def word2tri(words):
    count=Counter()
    for w in tqdm(words):
        count.update(explode_word(w))
    count = [["###", -1]] + count.most_common(tot_vocab - 1)
    dictionary = defaultdict(lambda: 0)
    for key, val in enumerate(count):
        dictionary[val[0]] = key
    return dictionary.keys()

trigram = word2tri(chi_wordset)
print(trigram)

def generate_batch():
    global data_index
    buffer_x = [] # batch_size X vocab_size
    buffer_y = [] # batch_size X positive_samples
    for _ in range(words_per_batch):
        # Filling in this particular x
        x = [0] * tot_vocab
        words = chi_wordset[data_index + half_window]
        trigram = word2tri(words)

        for i in range(len(trigram)):
            x[i] = 1
        
        # And sampling several y's:
        for _ in range(num_skips):
            y = [0] * positive_samples
            
            sample_word = rng.randint(-half_window, half_window - 1)
            if half_window >= 0: 
                sample_word += 1 # No zeroes allowed!
            
            sample_tris = list(word2tri(chi_wordset[data_index + half_window + sample_word]))
            for i, k in enumerate(rng.choice(sample_tris, positive_samples)):
                y[i] = k
            
            buffer_x.append(x)
            buffer_y.append(y)
        
        data_index = (data_index + 1) % (tot_vocab - window_size)
    return buffer_x, buffer_y
    

X, Y = generate_batch()
for i,k in enumerate(X[0]):
    if k!=0: print(i)

## train dssm model
# Running the model

np.set_printoptions(precision=3)
# saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(training_epochs):
        cc = []
        for _ in range(round((tot_vocab - window_size) / words_per_batch)):
            batch_x, batch_y = generate_batch()
            print(batch_x, batch_y)
            _, c = sess.run([optimizer, cost], feed_dict={layer_input: batch_x, layer_output_nums: batch_y})
            cc.append(c)
        print(cc)
#         if epoch % display_step == 0:
#             print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(np.mean(np.array(cc))))



