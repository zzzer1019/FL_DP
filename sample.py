# ==============================
# A sample that can run.
# ==============================

import tensorflow as tf
import mnist_inference as mnist
import Federated_Learning
from MNIST_reader import Data
import argparse
import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'




def sample(N, b, e, m, sigma, eps, save_dir, log_dir):

    hidden1 = 600
    hidden2 = 400

    DATA = Data(save_dir, N)

    with tf.Graph().as_default():

        train_op, eval_correct, loss, data_placeholder, labels_placeholder = mnist.mnist_fully_connected_model(b, hidden1, hidden2)

        Federated_Learning.run_federated_learning(loss, train_op, eval_correct, DATA, data_placeholder,
                                                      labels_placeholder, b=b, e=e, m=m, sigma=sigma, eps=eps,
                                                      save_dir=save_dir, log_dir=log_dir)


def main(_):
    sample(N=FLAGS.N, b=FLAGS.b, e=FLAGS.e,m=FLAGS.m, sigma=FLAGS.sigma, eps=FLAGS.eps, save_dir=FLAGS.save_dir, log_dir=FLAGS.log_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--save_dir',
        type=str,
        default=os.getcwd(),
        help='directory to store progress'
    )
    parser.add_argument(
        '--N',
        type=int,
        default=100,
        help='Total Number of clients participating'
    )
    parser.add_argument(
        '--sigma',
        type=float,
        default=0,
        help='The gm variance parameter; will not affect if Priv_agent is set to True'
    )
    parser.add_argument(
        '--eps',
        type=float,
        default=8,
        help='Epsilon'
    )
    parser.add_argument(
        '--m',
        type=int,
        default=0,
        help='Number of clients participating in a round'
    )
    parser.add_argument(
        '--b',
        type=float,
        default=10,
        help='Batches per client'
    )
    parser.add_argument(
        '--e',
        type=int,
        default=4,
        help='Epochs per client'
    )
    parser.add_argument(
        '--log_dir',
        type=str,
        default=os.path.join(os.getenv('TEST_TMPDIR', '/tmp'), 'tensorflow/mnist/logs/fully_connected_feed'),
        help='Directory to put the log data.'
    )

    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
