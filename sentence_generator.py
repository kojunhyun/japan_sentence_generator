# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import numpy as np
import tensorflow as tf

import corpus_reader as reader
tf.set_random_seed(777)

flags = tf.flags
logging = tf.logging

flags.DEFINE_string("data_path", None, "data_path")
flags.DEFINE_string("save_path", "ckpt", "checkpoint_dir")
flags.DEFINE_bool("use_fp16", False,
                  "Train using 16-bit floats instead of 32bit floats")
flags.DEFINE_bool("train", True, "should we train or test")

FLAGS = flags.FLAGS

def data_type():
    return tf.float16 if FLAGS.use_fp16 else tf.float32


class TrConfig(object):
    """Small config."""
    init_scale = 0.1
    learning_rate = 1.0
    max_grad_norm = 5
    num_layers = 2
    num_steps = 21
    hidden_size = 30
    max_epoch = 4
    max_max_epoch = 20
    keep_prob = 1.0
    lr_decay = 0.5
    batch_size = 64
    vocab_size = 4000


def get_config():
    return TrConfig()


class LSTM_Model(object):
    def __init__(self, is_training, config):
        self.batch_size = batch_size = config.batch_size
        self.num_steps = num_steps = config.num_steps
        size = config.hidden_size
        vocab_size = config.vocab_size

        self._input_data = tf.placeholder(tf.int32, [batch_size, num_steps])
        self._targets = tf.placeholder(tf.int32, [batch_size, num_steps])

        with tf.device("/cpu:0"):
            embedding = tf.get_variable("embedding", [vocab_size, size], dtype=data_type())
            inputs = tf.nn.embedding_lookup(embedding, self._input_data)

        if is_training and config.keep_prob < 1:
            inputs = tf.nn.dropout(inputs, config.keep_prob)

        output, state = self._build_rnn_graph_lstm(inputs, config, is_training)

        softmax_w = tf.get_variable("softmax_w", [size, vocab_size], dtype=data_type())
        softmax_b = tf.get_variable("softmax_b", [vocab_size], dtype=data_type())
        logits = tf.matmul(output, softmax_w) + softmax_b
        #logits = tf.nn.xw_plus_b(output, softmax_w, softmax_b)

        # Reshape logits to be a 3-D tensor for sequence loss
        logits = tf.reshape(logits, [self.batch_size, self.num_steps, vocab_size])
        self._logits = logits
        
        # Use the contrib sequence loss and average over the batches
        # many to many
        loss = tf.contrib.seq2seq.sequence_loss(
            logits,
            self._targets, ##
            tf.ones([self.batch_size, self.num_steps], dtype=data_type()),
            average_across_timesteps=False,
            average_across_batch=True)

        # Update the cost
        self._cost = tf.reduce_sum(loss)
        self._final_state = state

        # get the prediction accuracy
        self._softmax_out = tf.nn.softmax(tf.reshape(logits, [-1, vocab_size]))

        # shape는 [batch_size*num_steps, vocab_size] 이므로, y축을 기준으로 argmax를 취합니다.
        self._predict = tf.cast(tf.argmax(self._softmax_out, axis=1), tf.int32)

        
        self._correct_prediction = tf.equal(self._predict, tf.reshape(self._targets, [-1]))
        self._accuracy = tf.reduce_mean(tf.cast(self._correct_prediction, tf.float32))

        if not is_training:
            return

        self._lr = tf.Variable(0.0, trainable=False)
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self._cost, tvars), config.max_grad_norm)
        
        # optimizer 는 Adam 보다 gradientDescent 가 성능이 더 높게 측정...
        # Adam으로 수행할 때 지역극소에 빠지는듯...
        optimizer = tf.train.GradientDescentOptimizer(self._lr)
        #optimizer = tf.train.AdamOptimizer(self._lr)
        
        self._train_op = optimizer.apply_gradients(zip(grads, tvars), global_step=tf.contrib.framework.get_or_create_global_step())

        self._new_lr = tf.placeholder(tf.float32, shape=[], name="new_learning_rate")
        self._lr_update = tf.assign(self._lr, self._new_lr)

    
    def _get_lstm_cell(self, config, is_training):
        return tf.contrib.rnn.LSTMBlockCell(config.hidden_size, forget_bias=0.0)


    def _build_rnn_graph_lstm(self, inputs, config, is_training):
        """Build the inference graph using canonical LSTM cells."""
        # Slightly better results can be obtained with forget gate biases
        # initialized to 1 but the hyperparameters of the model would need to be
        # different than reported in the paper.
        cell = self._get_lstm_cell(config, is_training)
        if is_training and config.keep_prob < 1:
            cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=config.keep_prob)

        cell = tf.contrib.rnn.MultiRNNCell([cell for _ in range(config.num_layers)], state_is_tuple=True)
        self._initial_state = cell.zero_state(config.batch_size, data_type())
        state = self._initial_state
        # Simplified version of tensorflow_models/tutorials/rnn/rnn.py's rnn().
        # This builds an unrolled LSTM for tutorial purposes only.
        # In general, use the rnn() or state_saving_rnn() from rnn.py.
        #
        # The alternative version of the code below is:
        #
        # inputs = tf.unstack(inputs, num=num_steps, axis=1)
        # outputs, state = tf.contrib.rnn.static_rnn(cell, inputs,
        #                            initial_state=self._initial_state)
        outputs = []
        with tf.variable_scope("RNN"):
            for time_step in range(self.num_steps):
                if time_step > 0: tf.get_variable_scope().reuse_variables()
                (cell_output, state) = cell(inputs[:, time_step, :], state)
                outputs.append(cell_output)
        output = tf.reshape(tf.concat(outputs, 1), [-1, config.hidden_size])
        return output, state

    
    def assign_lr(self, session, lr_value):
        session.run(self._lr_update, feed_dict={self._new_lr: lr_value})

    @property
    def input_data(self):
        return self._input_data

    @property
    def targets(self):
        return self._targets
    
    @property
    def initial_state(self):
        return self._initial_state
    
    @property
    def final_state(self):
        return self._final_state

    @property
    def logits(self):
        return self._logits

    @property
    def lr(self):
        return self._lr

    @property
    def correct_prediction(self):
        return self._correct_prediction

    @property
    def predict(self):
        return self._predict
    
    @property
    def accuracy(self):
        return self._accuracy

    @property
    def train_op(self):
        return self._train_op

    @property
    def cost(self):
        return self._cost


def run_epoch(session, model, inverseDictionary, data, eval_op, verbose=False):
    print('run_epoch')
    epoch_size = ((len(data) // model.batch_size) - 1) // model.num_steps
    start_time = time.time()
    costs = 0.0
    accs = 0.0
    iters = 0
    state = session.run(model.initial_state)

    for step, (x, y) in enumerate(reader.corpus_iterator(data, model.batch_size, model.num_steps)):

        fetches = [model.cost, model.final_state, model.logits, model.correct_prediction, model.accuracy, eval_op]

        feed_dict = {}
        feed_dict[model.input_data] = x
        feed_dict[model.targets] = y
        

        for i, (c, h) in enumerate(model.initial_state):
            feed_dict[c] = state[i].c
            feed_dict[h] = state[i].h

        cost, state, logits, eq, acc, _= session.run(fetches, feed_dict)

        costs += cost
        accs += acc
        iters += model.num_steps

         
        if verbose and step % (epoch_size // 10) == 10:
            print("%.3f perplexity: %.3f speed: %.0f wps" % (step * 1.0 / epoch_size, np.exp(costs / iters),
            iters * model.batch_size / (time.time() - start_time)))
            print('accuracy : ', accs/(step+1))
        
    return np.exp(costs / iters) , accs/(step+1)



def main(_):
    if not FLAGS.data_path:
        raise ValueError("Must set --data_path to data file path")
    
    train_data, valid_data, voca_size, char_to_id = reader.corpus_raw_data(FLAGS.data_path, FLAGS.save_path)

    # output을 역변환
    inverseDictionary = dict(zip(char_to_id.values(), char_to_id.keys()))

    config = get_config()
    #config.vocab_size = voca_size # change, 상위 voca만 사용하는게 아니라 전체를 사용하고 싶을 때

    with tf.Graph().as_default(), tf.Session() as session:
        initializer = tf.random_uniform_initializer(-config.init_scale, config.init_scale)
        with tf.variable_scope("model", reuse=None, initializer=initializer):
            m = LSTM_Model(is_training=True, config=config)
            tf.summary.scalar("Training Loss", m.cost)
            tf.summary.scalar("Training accuracy", m.accuracy)
            tf.summary.scalar("Learning Rate", m.lr)
        
        with tf.variable_scope("model", reuse=True, initializer=initializer):
            mvalid = LSTM_Model(is_training=False, config=config)
            tf.summary.scalar("Validation Loss", mvalid.cost)
            tf.summary.scalar("Training accuracy", mvalid.accuracy)
        
        saver = tf.train.Saver()

        tf.global_variables_initializer().run()
        if FLAGS.train: 
            print ('training')
            with open(FLAGS.save_path + 'result.txt', 'w', encoding='utf-8') as f:

                for i in range(config.max_max_epoch):

                    print('epoch : ', i)
                    lr_decay = config.lr_decay ** max(i - config.max_epoch, 0.0)
                    m.assign_lr(session, config.learning_rate * lr_decay)
                    
                    print("Epoch: %d Learning rate: %.3f" % (i + 1, session.run(m.lr)))
                    train_perplexity, train_accuracy = run_epoch(session, m, inverseDictionary, train_data, m.train_op, verbose=True)
                    print("Epoch: %d Train Perplexity: %.3f Train Accuracy: %.3f" % (i + 1, train_perplexity, train_accuracy))
                    valid_perplexity, valid_accuracy  = run_epoch(session, mvalid, inverseDictionary, valid_data, tf.no_op())
                    print("Epoch: %d Valid Perplexity: %.3f Valid Accuracy: %.3f" % (i + 1, valid_perplexity, valid_accuracy))

                    saver.save(session, FLAGS.save_path + 'model', global_step=i+1)

        else:
            print('testing')
            ckpt = tf.train.get_checkpoint_state(FLAGS.save_path)
            print(FLAGS.save_path )
            print (ckpt)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(session, ckpt.model_checkpoint_path)
            else:
                print ("No checkpoint file found")
            

    


if __name__ == "__main__":
    tf.app.run()
