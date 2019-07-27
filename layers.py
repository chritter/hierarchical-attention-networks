import tensorflow as tf
import numpy as np
from utils import get_shape

try:
  from tensorflow.contrib.rnn import LSTMStateTuple
except ImportError:
  LSTMStateTuple = tf.nn.rnn_cell.LSTMStateTuple



def bidirectional_rnn(cell_fw, cell_bw, inputs, input_lengths,
                      initial_state_fw=None, initial_state_bw=None,
                      scope=None):
  '''
  Creates a dynamic version of bidirectional RNN. Here inputs are batches of sentences.
  :param cell_fw:
  :param cell_bw:
  :param inputs: dim (number_sentences, word sequence, embedding), number_sentenes = batch size (not equal to model input batchsize!)
  :param input_lengths: sequence length of each sentence
  :param initial_state_fw:
  :param initial_state_bw:
  :param scope:
  :return:
  '''
  with tf.variable_scope(scope or 'bi_rnn') as scope:
    # Creates a dynamic version of bidirectional recurrent neural network.
    # initial_state_fw: (optional) An initial state for the forward RNN.
    #     This must be a tensor of appropriate type and shape
    #     `[batch_size, cell_fw.state_size]`.
    # inputs: this must be a tensor of shape: `[batch_size, word sequence, ...]`,
    # input_lengths: containing the actual lengths for each of the sequences in the batch.
    # fw_outputs: [batch_size, max_time, cell_fw.output_size/self.cell_dim]
    # *_state: containing the forward and the backward final states of bidirectional rnn. this is not what we want
    (fw_outputs, bw_outputs), (fw_state, bw_state) = tf.nn.bidirectional_dynamic_rnn(
      cell_fw=cell_fw,
      cell_bw=cell_bw,
      inputs=inputs,
      sequence_length=input_lengths, #the network is fully unrolled for the given lengths of the sequences
      initial_state_fw=initial_state_fw,
      initial_state_bw=initial_state_bw,
      dtype=tf.float32,
      scope=scope
    )
    # concatunating fw and bw outputs
    outputs = tf.concat((fw_outputs, bw_outputs), axis=2)

    # concatunating of states happends differently, depending on type
    def concatenate_state(fw_state, bw_state):
      if isinstance(fw_state, LSTMStateTuple):
        state_c = tf.concat(
          (fw_state.c, bw_state.c), 1, name='bidirectional_concat_c')
        state_h = tf.concat(
          (fw_state.h, bw_state.h), 1, name='bidirectional_concat_h')
        state = LSTMStateTuple(c=state_c, h=state_h)
        return state
      elif isinstance(fw_state, tf.Tensor):
        state = tf.concat((fw_state, bw_state), 1,
                          name='bidirectional_concat')
        return state
      elif (isinstance(fw_state, tuple) and
            isinstance(bw_state, tuple) and
            len(fw_state) == len(bw_state)):
        # multilayer
        state = tuple(concatenate_state(fw, bw)
                      for fw, bw in zip(fw_state, bw_state))
        return state

      else:
        raise ValueError(
          'unknown state type: {}'.format((fw_state, bw_state)))

    state = concatenate_state(fw_state, bw_state)

    return outputs, state


def masking(scores, sequence_lengths, score_mask_value=tf.constant(-np.inf)):
  '''
  Masks all values beyond sequence length as -inf.
  :param scores:
  :param sequence_lengths:
  :param score_mask_value:
  :return:
  '''
  # for each sequence create masks which are true in sequence until sequence_length, then false until maxlen
  score_mask = tf.sequence_mask(sequence_lengths, maxlen=tf.shape(scores)[1])

  # matrix of size scores with -inf values
  score_mask_values = score_mask_value * tf.ones_like(scores)

  # returns scores or the -inf values for score_mask is true or false
  return tf.where(score_mask, scores, score_mask_values)


def attention(inputs, att_dim, sequence_lengths, scope=None):
  '''

  :param inputs: input of dim [number sentences, number words, hidden state cell_dim]
  :param att_dim:
  :param sequence_lengths:
  :param scope:
  :return:
  '''

  # why this assert here?
  assert len(inputs.get_shape()) == 3 and inputs.get_shape()[-1].value is not None

  with tf.variable_scope(scope or 'attention'):

    # word-level context vector word_att_W, word context vector is random uniform initialized as in paper
    word_att_W = tf.get_variable(name='att_W', shape=[att_dim, 1])

    # 1 layer MLP as described in the paper, incl. tanh (eq5), att_dim = units
    # returns (num sent, num words, attention dim) or (collapsed dim, attendion_dim)
    projection = tf.layers.dense(inputs, att_dim, tf.nn.tanh, name='projection')

    # u_it (words, att dim) * word_att_W (att_dim,1) leads to alpha shape (words)
    alpha = tf.matmul(tf.reshape(projection, shape=[-1, att_dim]), word_att_W)

    # reshape to (num sentences?, num_words)
    alpha = tf.reshape(alpha, shape=[-1, get_shape(inputs)[1]])

    # mask each sentence with -1e15 if longer than what is given in sequence_lengths
    alpha = masking(alpha, sequence_lengths, tf.constant(-1e15, dtype=tf.float32))

    # apply softmax to calculate alphas as defined in paper
    alpha = tf.nn.softmax(alpha)

    # calculate s_i for each sentence: sum across number of words, for each sentence,
    # add expand dims due to hidden state cell dim in inputs
    # outputs dim (num sentences,1,hidden state cell dim)
    outputs = tf.reduce_sum(inputs * tf.expand_dims(alpha, 2), axis=1)

    return outputs, alpha
