

import tensorflow as tf
from datetime import datetime
from data_reader import DataReader
from model import Model
from utils import read_vocab, count_parameters, load_glove
import sys
import mlflow
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment('EventDetection')
from mlflow import tensorflow
tensorflow.autolog(every_n_iter=100) #default 100

# Parameters
# ==================================================
FLAGS = tf.flags.FLAGS

# ----------------------- Setup

# input trian data

tf.flags.DEFINE_string("vocab","data/pre_pro_imdb/imdb-w2i.pkl",
                       """Path to file with vocabulary""")
tf.flags.DEFINE_string("train_data_file","data/pre_pro_imdb/imdb-train.pkl",
                       """Path to training data pickled""")
tf.flags.DEFINE_string("dev_data_file","data/pre_pro_imdb/imdb-dev.pkl",
                       """Path to development data pickled""")
tf.flags.DEFINE_string("test_data_file","data/pre_pro_imdb/imdb-test.pkl",
                       """Path to test data pickled""")

# select the word embeddings, must agree with the embedding size emb_size flag
tf.flags.DEFINE_string("embedding_file","../WordEmbeddings/Data/glove.6B/glove.6B.200d.txt",
                       """path to file with embeddings""")

tf.flags.DEFINE_string("checkpoint_dir", 'saved_models/run2/checkpoints',
                       """Path to checkpoint folder""")
tf.flags.DEFINE_string("log_dir", 'saved_models/run2/logs',
                       """Path to log folder""")

tf.flags.DEFINE_integer("display_step", 20,
                        """Number of steps to display log into TensorBoard (default: 20)""")

#if dataset =='yelp15':
#    tf.flags.DEFINE_integer("num_classes", 5,
#                            """Number of classes (default: 5)""")
tf.flags.DEFINE_integer("num_classes", 10, #default imdb data set setting
                            """Number of classes (default: 5)""")

tf.flags.DEFINE_integer("num_checkpoints", 1,
                        """Number of checkpoints to store (default: 1)""")

tf.flags.DEFINE_boolean("allow_soft_placement", True,
                        """Allow device soft device placement""")


# ----------------------- Hyperparameter

## Network structure

tf.flags.DEFINE_integer("cell_dim", 50,
                        """Hidden dimensions of GRU cells (default: 50)""")
tf.flags.DEFINE_integer("att_dim", 100,
                        """Dimensionality of attention spaces (default: 100)""")
tf.flags.DEFINE_integer("emb_size", 200,
                        """Dimensionality of word embedding (default: 200)""")

## Training//learning parameters

tf.flags.DEFINE_integer("num_epochs", 20,
                        """Number of training epochs (default: 20)""")
tf.flags.DEFINE_integer("batch_size", 64,
                        """Batch size (default: 64)""")
tf.flags.DEFINE_float("learning_rate", 0.0005,
                      """Learning rate (default: 0.0005)""")
# Clips values of multiple tensors by the ratio of the sum of their norms.
tf.flags.DEFINE_float("max_grad_norm", 5.0,
                      """Maximum value of the global norm of the gradients for clipping (default: 5.0)""")
tf.flags.DEFINE_float("dropout_rate", 0.5,
                      """Probability of dropping neurons (default: 0.5)""")

FLAGS = tf.flags.FLAGS
FLAGS(sys.argv, known_only=True)

# ==================================================

# define here user specific input FLAGS.** =

# <--------- run specific settings
FLAGS.checkpoint_dir = 'saved_models/run3/checkpoints'
FLAGS.log_dir = 'saved_models/run3/log'
FLAGS.num_classes = 10
FLAGS.vocab = 'data/pre_pro_imdb/imdb-w2i.pkl'
FLAGS.train_data_file = 'data/pre_pro_imdb_small/imdb-train.pkl'
FLAGS.dev_data_file =  'data/pre_pro_imdb_small/imdb-dev.pkl'
FLAGS.test_data_file =  'data/pre_pro_imdb_small/imdb-test.pkl'




for key, values in FLAGS.flag_values_dict().items():
    mlflow.log_param(key,values)

# True if the path exists, whether it's a file or a directory. F
if not tf.gfile.Exists(FLAGS.checkpoint_dir):
  tf.gfile.MakeDirs(FLAGS.checkpoint_dir)

if not tf.gfile.Exists(FLAGS.log_dir):
  tf.gfile.MakeDirs(FLAGS.log_dir)

# summary writers for metrics (accuracy etc.)
train_writer = tf.summary.FileWriter(FLAGS.log_dir + '/train')
valid_writer = tf.summary.FileWriter(FLAGS.log_dir + '/valid')
test_writer = tf.summary.FileWriter(FLAGS.log_dir + '/test')



def loss_fn(labels, logits):
  '''
  Calculate the cross-entropy loss based on labels and logits input
  '''
  # one-hot encode labels
  onehot_labels = tf.one_hot(labels, depth=FLAGS.num_classes)
  # Yang16 uses log likelyhood equivalent to the cross entropy used here
  cross_entropy_loss = tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels,
                                                       logits=logits)
  tf.summary.scalar('loss', cross_entropy_loss)
  return cross_entropy_loss


def train_fn(loss):
  '''
  Calculate gradients and update parameters based on loss
  '''
  # get all trainaible variables
  trained_vars = tf.trainable_variables()

  # great utils function to calculate the parameters of the model and print them out
  count_parameters(trained_vars)

  # get gradients for parameter given loss
  gradients = tf.gradients(loss, trained_vars)

  # Gradient clipping (described in paper?): Clips values of multiple tensors by the ratio of the sum of their norms.
  clipped_grads, global_norm = tf.clip_by_global_norm(gradients, FLAGS.max_grad_norm)
  # save global norm
  tf.summary.scalar('global_grad_norm', global_norm)

  # Add gradients and vars to summary
  # for gradient, var in list(zip(clipped_grads, trained_vars)):
  #   if 'attention' in var.name:
  #     tf.summary.histogram(var.name + '/gradient', gradient)
  #     tf.summary.histogram(var.name, var)

  # Define optimizer
  # Returns and create (if necessary) the global step tensor.
  global_step = tf.train.get_or_create_global_step()
  # define the optimizer rmsprop; paper uses different optimizer: SGD with momentum 0.9
  optimizer = tf.train.RMSPropOptimizer(FLAGS.learning_rate)
  # get apply gradients operation
  train_op = optimizer.apply_gradients(zip(clipped_grads, trained_vars),
                                       name='train_op',
                                       global_step=global_step)
  return train_op, global_step


def eval_fn(labels, logits):
  '''
  Calculates average batch accuracy, overall accuracy and returns some operations
  '''
  # get index of best predictions
  predictions = tf.argmax(logits, axis=-1)
  # check agreement between prediction and labels (as integer)
  correct_preds = tf.equal(predictions, tf.cast(labels, tf.int64))
  # calcualte average accuracy of batch data point
  batch_acc = tf.reduce_mean(tf.cast(correct_preds, tf.float32))
  # save accuracy
  tf.summary.scalar('accuracy', batch_acc)

  # calculates overall accuracy
  # acc_update: An operation that increments the total and count variables appropriately and whose value matches accuracy
  total_acc, acc_update = tf.metrics.accuracy(labels, predictions, name='metrics/acc')
  # intersting: we get all variables related to metrics scope and initialize
  metrics_vars = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="metrics")
  metrics_init = tf.variables_initializer(var_list=metrics_vars)

  return batch_acc, total_acc, acc_update, metrics_init


def main(_):

    # load the word_to_index encoded vocabulary
    vocab = read_vocab(FLAGS.vocab)

    # create embedding matrix of size (vocab,emb_size)
    glove_embs = load_glove(FLAGS.embedding_file,FLAGS.emb_size, vocab)

    print('input embeddings shape: ',glove_embs.shape)

    # read data
    data_reader = DataReader(train_file=FLAGS.train_data_file,
                             dev_file=FLAGS.dev_data_file,
                             test_file=FLAGS.test_data_file,
                             num_classes=FLAGS.num_classes)

    config = tf.ConfigProto(allow_soft_placement=FLAGS.allow_soft_placement)

    tf.reset_default_graph()

    sess = tf.Session(config=config)

    model = Model(cell_dim=FLAGS.cell_dim,
                  att_dim=FLAGS.att_dim,
                  vocab_size=len(vocab),
                  emb_size=FLAGS.emb_size,
                  num_classes=FLAGS.num_classes,
                  dropout_rate=FLAGS.dropout_rate,
                  pretrained_embs=glove_embs)

    # calculate loss
    loss = loss_fn(model.labels, model.logits)

    # calculates gradients
    train_op, global_step = train_fn(loss)

    # calculates metrics and merges all
    batch_acc, total_acc, acc_update, metrics_init = eval_fn(model.labels, model.logits)
    summary_op = tf.summary.merge_all()

    sess.run(tf.global_variables_initializer())

    # The graph described by sess.graph will be displayed by TensorBoard
    train_writer.add_graph(sess.graph)

    # save all variables
    saver = tf.train.Saver(max_to_keep=FLAGS.num_checkpoints)

    print('\n{}> Start training'.format(datetime.now()))

    epoch = 0
    valid_step = 0
    test_step = 0
    train_test_prop = len(data_reader.train_data) / len(data_reader.test_data)

    test_batch_size = int(FLAGS.batch_size / train_test_prop)
    best_acc = float('-inf')

    while epoch < FLAGS.num_epochs:
      epoch += 1
      print('\n{}> Epoch: {}'.format(datetime.now(), epoch))

      # we newly initialize metrics tensors each epoch, each evaluation
      sess.run(metrics_init)

      # each data point/doc in batch contains a list of sentences, encoded with index
      for batch_docs, batch_labels in data_reader.read_train_set(FLAGS.batch_size, shuffle=True):

        # do a batch
        _step, _, _loss, _acc, _ = sess.run([global_step, train_op, loss, batch_acc, acc_update],
                                         feed_dict=model.get_feed_dict(batch_docs, batch_labels, training=True))

        # each display_step steps evaluate metric variables and add to train_writer, training is false to disables dropout
        if _step % FLAGS.display_step == 0: #
          _summary = sess.run(summary_op, feed_dict=model.get_feed_dict(batch_docs, batch_labels))
          train_writer.add_summary(_summary, global_step=_step)
      # evaluate batch accuracy and print
      print('Training accuracy = {:.2f}'.format(sess.run(total_acc) * 100))
      last_step_epoch = _step

      # we newly initialize metrics tensors each epoch, each evaluation
      sess.run(metrics_init)

      # for each epoch calculate metrics for valid set
      valid_loss = 0.
      valid_acc = 0.
      for batch_docs, batch_labels in data_reader.read_valid_set(test_batch_size):
        _loss, _acc, _  = sess.run([loss, batch_acc, acc_update], feed_dict=model.get_feed_dict(batch_docs, batch_labels))
        valid_loss += _loss
        valid_acc += _acc
        valid_step += 1
        #if valid_step % FLAGS.display_step == 0:
        #_summary = sess.run(summary_op, feed_dict=model.get_feed_dict(batch_docs, batch_labels))
      valid_loss = valid_loss/valid_step
      valid_acc = valid_acc/valid_step
      summary_loss = tf.Summary(value=[tf.Summary.Value(tag='valid_loss', simple_value=valid_loss)])
      valid_writer.add_summary(summary_loss,global_step=last_step_epoch)
      summary_acc = tf.Summary(value=[tf.Summary.Value(tag='valid_accuracy', simple_value=valid_acc)])
      valid_writer.add_summary(summary_acc,global_step=last_step_epoch)
      #valid_writer.add_summary(_summary, global_step=valid_step)
      print('Validation accuracy = {:.2f}  and {:.2f}  '.format(sess.run(total_acc) * 100,valid_acc))

      # we newly initialize metrics tensors each epoch, each evaluation
      sess.run(metrics_init)

      # for each epoch calculate metrics for test set
      test_loss = 0.
      test_acc = 0.
      for batch_docs, batch_labels in data_reader.read_test_set(test_batch_size):
        _loss, _acc, _  = sess.run([loss, batch_acc, acc_update], feed_dict=model.get_feed_dict(batch_docs, batch_labels))
        test_loss += _loss
        test_acc += _acc
        test_step += 1
        #if test_step % FLAGS.display_step == 0:
        #  _summary = sess.run(summary_op, feed_dict=model.get_feed_dict(batch_docs, batch_labels))
        #  test_writer.add_summary(_summary, global_step=test_step)
      test_loss = test_loss/test_step
      test_acc = test_acc/test_step
      summary_loss = tf.Summary(value=[tf.Summary.Value(tag='test_loss', simple_value=test_loss)])
      test_writer.add_summary(summary_loss,global_step=last_step_epoch)
      summary_acc = tf.Summary(value=[tf.Summary.Value(tag='test_accuracy', simple_value=test_acc)])
      test_writer.add_summary(summary_acc,global_step=last_step_epoch)
      #valid_writer.add_summary(_summary, global_step=valid_step)
      print('Test accuracy = {:.2f}  and {:.2f}  '.format(sess.run(total_acc) * 100,test_acc))



      #test_acc = sess.run(total_acc) * 100
      print('Testing accuracy = {:.2f}'.format(test_acc))

      # keep track of best test accuracy, if epoch improved, save all variables
      if test_acc > best_acc:
        best_acc = test_acc
        saver.save(sess, FLAGS.checkpoint_dir)
      print('Best testing accuracy = {:.2f}'.format(test_acc))

      print("{} Optimization Finished!".format(datetime.now()))
      print('Best testing accuracy = {:.2f}'.format(best_acc))


if __name__ == '__main__':
  tf.app.run()
