# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from flask import Flask, request, render_template, Response
import json

flags = tf.flags
logging = tf.logging
## CPU mode, 제거하면 GPU mode
t_config = tf.ConfigProto(device_count={'GPU':0}) 
################################################
flags.DEFINE_string("vocab_path", None, "vocalbulary txt data path")
flags.DEFINE_string("save_path", "ckpt", "checkpoint_dir")
flags.DEFINE_integer("port_num", 33002, "flask port number")
flags.DEFINE_bool("use_fp16", False, "Train using 16-bit floats instead of 32bit floats")
FLAGS = flags.FLAGS

flask_app = Flask(__name__)

#@flask_app.route('/', methods=['GET'])
#def hello_tensorflow():
#    return 'Hello tensorflow!'

def data_type():
    return tf.float16 if FLAGS.use_fp16 else tf.float32


class TeConfig(object):
    """Small config."""
    init_scale = 0.1
    learning_rate = 1.0
    max_grad_norm = 5
    num_layers = 2
    num_steps = 1
    hidden_size = 30
    max_epoch = 4
    max_max_epoch = 20
    keep_prob = 1.0
    lr_decay = 0.5
    batch_size = 1
    vocab_size = 10000 # 의미 없음


class LSTM_Model(object):
    def __init__(self, is_training, config):
        self.batch_size = batch_size = config.batch_size
        self.num_steps = num_steps = config.num_steps
        size = config.hidden_size
        vocab_size = config.vocab_size

        self._input_data = tf.placeholder(tf.int32, [batch_size, num_steps])
        self._targets = tf.placeholder(tf.int32, [batch_size, num_steps])

        print('model start')
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
        self._predict = tf.cast(tf.argmax(self._softmax_out, axis=1), tf.int32)
        self._correct_prediction = tf.equal(self._predict, tf.reshape(self._targets, [-1]))
        self._accuracy = tf.reduce_mean(tf.cast(self._correct_prediction, tf.float32))


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
    def softmax_out(self):
        return self._softmax_out

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
    def initial_state(self):
        return self._initial_state

    @property
    def cost(self):
        return self._cost

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
    def train_op(self):
        return self._train_op

    @property
    def initial_state_name(self):
        return self._initial_state_name

    @property
    def final_state_name(self):
        return self._final_state_name


def multiTest(input):
    uni_word = input
    model = mtest
    word_to_idx = word_to_id
    recommend_num = 10

    """Runs the real time test"""
    state = session.run(model.initial_state)

    virtual_word = ''
    y_sentences = ''
    results_sentences = []
    word_idx = np.array(1).reshape(1,1)
    

    fetches = [model.final_state, model.logits, eval_op]
    
    for char in uni_word:
        feed_dict = {}
        if not char in word_to_idx:
            word_idx[0] = word_to_idx['<unk>']
        else:
            word_idx[0] = word_to_idx[char]
        feed_dict[model.input_data] = word_idx
        for i, (c, h) in enumerate(model.initial_state):
            feed_dict[c] = state[i].c
            feed_dict[h] = state[i].h
        state, logits, _ = session.run(fetches, feed_dict)

    logits_output = dict(zip(logits[0][0], range(len(logits[0][0]))))
    logits_result = sorted(logits_output.items(), reverse=True)
    output_results = logits_result[:recommend_num]

    second_state = state
    
    #for output_dict in output_results:
    for i in range(0, recommend_num):
        #print(output_dict[1])
        word_idx[0] = output_results[i][1]
        
        #print(y_sentences)

        if inverseDictionary[output_results[i][1]] == '<eos>':
            y_sentences = uni_word
            results_sentences.append(y_sentences)
        else:
            y_sentences = uni_word + inverseDictionary[output_results[i][1]]
            state = second_state
            jug_eos = ''
            while jug_eos != '<eos>':
                fetches = [model.final_state, model.logits, eval_op]
                feed_dict = {}
                feed_dict[model.input_data] = word_idx

                for j, (c, h) in enumerate(model.initial_state):
                    feed_dict[c] = state[j].c
                    feed_dict[h] = state[j].h
                state, logits, _ = session.run(fetches, feed_dict)         

                decodedWordId = int(np.argmax(logits))
                virtual_word = inverseDictionary[decodedWordId]                
                

                if virtual_word == '<eos>':
                    jug_eos = '<eos>'
                else:
                    y_sentences = y_sentences + virtual_word

                word_idx[0] = decodedWordId
        
            results_sentences.append(y_sentences)
        print(y_sentences)
    result_output = []
    for i in range(len(results_sentences)):
        result_output.append({'output':results_sentences[i]})

    print(result_output)
    #return 'output : %s ' %results_sentences[0]
    #return render_template('output_print.html', data=results_sentences)
    return json.dumps(result_output, ensure_ascii=False)


@flask_app.route('/real_testing/<name>', methods=['GET', 'POST'])
#@flask_app.route('/<name>', methods=['GET', 'POST'])
def real_testing(name):
    print('real_test')

    output = multiTest(name)

    return output


def main(_):
    global session, mtest, word_to_id, inverseDictionary
    global eval_op

    with open(FLAGS.vocab_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    words = []
    for line in lines:
        word = line.split('\t')
        words.append(word[0])
        #words.append(word[0].decode('utf-8'))
    word_to_id = dict(zip(words, range(len(words))))
    print('len(word_to_id) : ', len(word_to_id))
    inverseDictionary = dict(zip(word_to_id.values(), word_to_id.keys()))
    

    eval_config = TeConfig()
    eval_config.vocab_size = len(word_to_id) # change
    

    with tf.Graph().as_default(), tf.Session(config=t_config) as session: 
        initializer = tf.random_uniform_initializer(-eval_config.init_scale, eval_config.init_scale)
        with tf.variable_scope("model", reuse=None, initializer=initializer):
            mtest = LSTM_Model(is_training=False, config=eval_config)
    

        saver = tf.train.Saver()
        tf.global_variables_initializer().run()

        print ('testing')
        ckpt = tf.train.get_checkpoint_state(FLAGS.save_path )
        print(ckpt)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(session, ckpt.model_checkpoint_path)
        else:
            return print("No checkpoint file found")
    
        eval_op = tf.no_op()

        flask_app.run(host="0.0.0.0", port=FLAGS.port_num)
    


if __name__ == "__main__":
    tf.app.run()
