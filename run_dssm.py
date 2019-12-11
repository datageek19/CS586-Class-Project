import time
import numpy as np
import random

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

random.seed(43)

# start = time.time()
query_BS = 500
num_layer_1 = 256
num_layer_2 = 256
learning_rate = 0.001
summaries_dir = './Summaries/'
num_epoch = 15

file_train = 'data/oppo_round1_train_20180929_mini.txt'
file_vali = 'data/oppo_round1_vali_20180929_mini.txt'
vocab_path = './data/vocab.txt'

def load_vocab(file_loc):
    word_dict = {}
    with open(file_loc, encoding='utf8') as f:
        for idx, word in enumerate(f.readlines()):
            word = word.strip()
            word_dict[word] = idx
    return word_dict

vocab_map = load_vocab(vocab_path)
tot_words = len(vocab_map)
unk = '[UNK]'
pad = '[PAD]'

def seq2bow(query, vocab_map):
    bow_ids = np.zeros(tot_words)
    for w in query:
        if w in vocab_map:
            bow_ids[vocab_map[w]] += 1
        else:
            bow_ids[vocab_map[unk]] += 1
    return bow_ids

def get_data_bow(file_loc):
    data_arr = []
    with open(file_loc, encoding='utf8') as f:
        for line in f.readlines():
            spline = line.strip().split('\t')
            if len(spline) < 4:
                continue
            prefix, _, title, tag, label = spline
            prefix_ids = seq2bow(prefix, vocab_map)
            title_ids = seq2bow(title, vocab_map)
            data_arr.append([prefix_ids, title_ids, int(label)])
    return data_arr


data_train =get_data_bow(file_train)
data_vali = get_data_bow(file_vali)

train_epoch_steps = int(len(data_train) / query_BS) - 1
vali_epoch_steps = int(len(data_vali) / query_BS) - 1

def batch_normalization(x, phase_train, out_size):
    with tf.variable_scope('bn'):
        beta = tf.Variable(tf.constant(0.0, shape=[out_size]),
                           name='beta', trainable=True)
        gamma = tf.Variable(tf.constant(1.0, shape=[out_size]),
                            name='gamma', trainable=True)
        batch_mean, batch_var = tf.nn.moments(x, [0], name='moments')
        ema = tf.train.ExponentialMovingAverage(decay=0.5)

        def mean_var_with_update():
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        mean, var = tf.cond(phase_train,
                            mean_var_with_update,
                            lambda: (ema.average(batch_mean), ema.average(batch_var)))
        normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
    return normed

def variable_summaries(var, name):
    """Attach a lot of summaries to a Tensor."""
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean/' + name, mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_sum(tf.square(var - mean)))
        tf.summary.scalar('sttdev/' + name, stddev)
        tf.summary.scalar('max/' + name, tf.reduce_max(var))
        tf.summary.scalar('min/' + name, tf.reduce_min(var))
        tf.summary.histogram(name, var)

with tf.name_scope('input'):
    query_batch = tf.placeholder(tf.float32, shape=[None, None], name='query_batch')
    doc_batch = tf.placeholder(tf.float32, shape=[None, None], name='doc_batch')
    doc_label_batch = tf.placeholder(tf.float32, shape=[None], name='doc_label_batch')
    on_train = tf.placeholder(tf.bool)
    keep_prob = tf.placeholder(tf.float32, name='drop_out_prob')

with tf.name_scope('FC1'):
    l1_par_range = np.sqrt(6.0 / (tot_words + num_layer_1))
    weight1 = tf.Variable(tf.random_uniform([tot_words, num_layer_1], -l1_par_range, l1_par_range))
    bias1 = tf.Variable(tf.random_uniform([num_layer_1], -l1_par_range, l1_par_range))
    variable_summaries(weight1, 'L1_weights')
    variable_summaries(bias1, 'L1_biases')
    query_l1 = tf.matmul(query_batch, weight1) + bias1
    doc_l1 = tf.matmul(doc_batch, weight1) + bias1
    query_l1_out = tf.nn.relu(query_l1)
    doc_l1_out = tf.nn.relu(doc_l1)
    
with tf.name_scope('BN1'):
    query_l1 = batch_normalization(query_l1_out, on_train, num_layer_1)
    doc_l1 = batch_normalization(doc_l1_out, on_train, num_layer_1)
    query_l1_out = tf.nn.relu(query_l1)
    doc_l1_out = tf.nn.relu(doc_l1)

with tf.name_scope('Drop_out'):
    query_l1_out = tf.nn.dropout(query_l1_out, keep_prob)
    doc_l1_out = tf.nn.dropout(doc_l1_out, keep_prob)

with tf.name_scope('FC2'):
    l2_par_range = np.sqrt(6.0 / (num_layer_1 + num_layer_2))
    weight2 = tf.Variable(tf.random_uniform([num_layer_1, num_layer_2], -l2_par_range, l2_par_range))
    bias2 = tf.Variable(tf.random_uniform([num_layer_2], -l2_par_range, l2_par_range))
    variable_summaries(weight2, 'L2_weights')
    variable_summaries(bias2, 'L2_biases')

    query_l2 = tf.matmul(query_l1_out, weight2) + bias2
    doc_l2 = tf.matmul(doc_l1_out, weight2) + bias2
    query_y = tf.nn.relu(query_l2)
    doc_y = tf.nn.relu(doc_l2)

with tf.name_scope('BN2'):
    query_l2 = batch_normalization(query_l2, on_train, num_layer_2)
    doc_l2 = batch_normalization(doc_l2, on_train, num_layer_2)
    query_l2 = tf.nn.relu(query_l2)
    doc_l2 = tf.nn.relu(doc_l2)
    query_pred = tf.nn.relu(query_l2)
    doc_pred = tf.nn.relu(doc_l2)

with tf.name_scope('Cosine_Similarity'):
    pooled_len_1 = tf.sqrt(tf.reduce_sum(tf.square(query_pred), 1))
    pooled_len_2 = tf.sqrt(tf.reduce_sum(tf.square(doc_pred), 1))
    pooled_mul_12 = tf.reduce_sum(tf.multiply(query_pred, doc_pred), 1)
    cos_scores = tf.div(pooled_mul_12, pooled_len_1 * pooled_len_2 + 1e-8, name="cos_scores")
    cos_sim_prob = tf.clip_by_value(cos_scores, 1e-8, 1.0)

with tf.name_scope('Loss'):
    # train loss
    cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=doc_label_batch, logits=cos_scores)
    losses = tf.reduce_sum(cross_entropy)
    tf.summary.scalar('loss', losses)
    pass

with tf.name_scope('Training'):
    # use SGD Optimizer
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(losses)

with tf.name_scope('Accuracy'):
    accuracy = tf.reduce_mean(tf.cast(cross_entropy, tf.float32))
    tf.summary.scalar('accuracy', accuracy)

with tf.name_scope('Test'):
    test_average_loss = tf.placeholder(tf.float32)
    test_avg_accuracy = tf.placeholder(tf.float32)
    test_loss_summary = tf.summary.scalar('test_average_loss', test_average_loss)
    test_accuracy_summary = tf.summary.scalar('test_avg_accuracy', test_avg_accuracy)

with tf.name_scope('Train'):
    train_average_loss = tf.placeholder(tf.float32)
    train_avg_accuracy = tf.placeholder(tf.float32)
    train_loss_summary = tf.summary.scalar('train_average_loss', train_average_loss)
    train_accuracy_summary = tf.summary.scalar('train_avg_accuracy', train_avg_accuracy)

def pull_batch(data_map, batch_id):
    query, title, label, dsize = range(4)
    cur_data = data_map[batch_id * query_BS:(batch_id + 1) * query_BS]
    query_in = [x[0] for x in cur_data]
    doc_in = [x[1] for x in cur_data]
    label = [x[2] for x in cur_data]
    return query_in, doc_in, label

def feed_dict(on_training, data_set, batch_id, drop_prob):
    query_in, doc_in, label = pull_batch(data_set, batch_id)
    query_in, doc_in, label = np.array(query_in), np.array(doc_in), np.array(label)
    return {query_batch: query_in, doc_batch: doc_in, doc_label_batch: label,
            on_train: on_training, keep_prob: drop_prob}

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    train_writer = tf.summary.FileWriter(summaries_dir + '/train', sess.graph)
    test_writer = tf.summary.FileWriter(summaries_dir+'/test', sess.graph)

    # start = time.time()
    for epoch in range(num_epoch):
        random.shuffle(data_train)
        for batch_id in range(train_epoch_steps):
            sess.run(train_step, feed_dict=feed_dict(True, data_train, batch_id, 0.5))
            pass
        # end = time.time()
        # train loss
        epoch_loss = 0
        for i in range(train_epoch_steps):
            loss_v = sess.run(losses, feed_dict=feed_dict(False, data_train, i, 1))
            epoch_loss += loss_v

        epoch_loss /= (train_epoch_steps)
        train_loss_ = sess.run(train_loss_summary, feed_dict={train_average_loss: epoch_loss})
        train_writer.add_summary(train_loss_, epoch + 1)
        print("\nEpoch #%d | Train Loss: %-4.3f " %
              (epoch, epoch_loss))

        # test loss
        epoch_loss = 0
        for i in range(vali_epoch_steps):
            loss_v = sess.run(losses, feed_dict=feed_dict(False, data_vali, i, 1))
            epoch_loss += loss_v
        epoch_loss /= (vali_epoch_steps)
        test_loss_ = sess.run(test_loss_summary, feed_dict={test_average_loss: epoch_loss})
        train_writer.add_summary(test_loss_, epoch + 1)
        # test_writer.add_summary(test_loss, step + 1)
        print("Epoch #%d | Test  Loss: %-4.3f " % (epoch, epoch_loss))

        ## train accuracy
        epoch_train_accu = 0
        for i in range(train_epoch_steps):
            train_accu_val = sess.run(accuracy, feed_dict=feed_dict(False, data_train, i, 1))
            epoch_train_accu += train_accu_val
        epoch_train_accu /= (train_epoch_steps)
        train_accuracy_ = sess.run(train_accuracy_summary, feed_dict={train_avg_accuracy: epoch_train_accu})
        train_writer.add_summary(train_accuracy_, epoch + 1)
        print("\nEpoch #%d | Train accuracy: %-4.3f " % (epoch, epoch_train_accu))

        ## test accuracy
        epoch_test_accu = 0
        for i in range(vali_epoch_steps):
            test_accu_val = sess.run(accuracy, feed_dict=feed_dict(False, data_vali, i, 1))
            epoch_test_accu += test_accu_val
        epoch_test_accu /= (vali_epoch_steps)
        test_accuracy = sess.run(test_accuracy_summary, feed_dict={test_avg_accuracy: epoch_test_accu})
        test_writer.add_summary(test_accuracy, epoch + 1)
        print("\nEpoch #%d | Test accuracy: %-4.3f " % (epoch, epoch_test_accu))