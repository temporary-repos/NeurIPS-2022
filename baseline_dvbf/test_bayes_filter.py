import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import tensorflow as tf
import numpy as np
from bayes_filter import BayesFilter
from replay_memory import BouncingBallData
import random
import scipy.io as sio


def main():
    ######################################
    #          General Params            #
    ######################################
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', type=str, default='bf_checkpoints',
                        help='directory to store checkpointed models')
    parser.add_argument('--val_frac', type=float, default=0.1, help='fraction of data to be witheld in validation set')
    parser.add_argument('--ckpt_name', type=str, default='ep99.ckpt-99', help='name of checkpoint file to load (blank means none)')
    parser.add_argument('--save_name', type=str, default='bf_model', help='name of checkpoint files for saving')
    parser.add_argument('--domain_name', type=str, default='Pendulum-v0', help='environment name')
    parser.add_argument('--device', type=str, default='0', help='which GPU to use')

    parser.add_argument('--exp_name', type=str, default='grav16', help='name of the experiment to save')
    parser.add_argument('--exp_id', type=str, default='1', help='id of the experiment')
    parser.add_argument('--exp_type', type=str, default='mixed_gravity_16_pred_1', help='type of the experiment, ie mixed')
    parser.add_argument('--test', type=int, default=0, help='type of the experiment, ie mixed')


    parser.add_argument('--seq_length', type=int, default=8, help='sequence length for training')
    parser.add_argument('--z_start_len', type=int, default=8, help='sequence length for training')

    parser.add_argument('--batch_size', type=int, default=32, help='minibatch size')
    parser.add_argument('--code_dim', type=int, default=4, help='dimensionality of code')
    parser.add_argument('--noise_dim', type=int, default=4, help='dimensionality of noise vector')
    parser.add_argument('--state_dim', type=int, default=32 * 32, help='dimensionality of state vector')

    parser.add_argument('--num_epochs', type=int, default=100, help='number of epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')
    parser.add_argument('--decay_rate', type=float, default=0.99, help='decay rate for learning rate')
    parser.add_argument('--grad_clip', type=float, default=5.0, help='clip gradients at this value')

    parser.add_argument('--n_trials', type=int, default=100, help='number of data sequences to collect in each episode')
    parser.add_argument('--trial_len', type=int, default=20, help='number of steps in each trial')
    parser.add_argument('--n_subseq', type=int, default=3, help='number of subsequences to divide each sequence into')

    parser.add_argument('--kl_weight', type=float, default=1,
                        help='weight applied to kl-divergence loss, annealed if zero')
    parser.add_argument('--start_kl', type=int, default=5,
                        help='epoch of training in which to start enforcing KL penalty')
    parser.add_argument('--anneal_time', type=int, default=25, help='number of epochs over which to anneal KLD')

    ######################################
    #    Feature Extractor Params        #
    ######################################
    parser.add_argument('--num_filters', nargs='+', type=int, default=[8],
                        help='number of filters after each down/uppconv')
    parser.add_argument('--reg_weight', type=float, default=1e-4, help='weight applied to regularization losses')
    parser.add_argument('--feature_dim', type=int, default=8, help='dimensionality of extracted features')

    ######################################
    #          Additional Params         #
    ######################################
    parser.add_argument('--rnn_size', type=int, default=100, help='size of rnn layer')
    parser.add_argument('--transform_size', type=int, default=64,
                        help='generic size of layers performing transformations')
    parser.add_argument('--extractor_size', nargs='+', type=int, default=[32],
                        help='hidden layer sizes in feature extractor/decoder')
    parser.add_argument('--num_matrices', type=int, default=8, help='number of matrices to be combined for propagation')
    parser.add_argument('--inference_size', nargs='+', type=int, default=[32], help='size of inference network')

    args = parser.parse_args()
    args.noise_dim = args.code_dim
    args.feature_dim = args.code_dim
    args.state_dim = 1024
    args.action_dim = 1

    # Set random seed
    random.seed(1)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    # Generate the dataset
    replay_memory = BouncingBallData(args, file_path='physics_data/' + args.exp_type)

    # Construct model
    tf.reset_default_graph()
    net = BayesFilter(args)
    test(args, net, replay_memory)


def test(args, net, replay_memory):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    # Begin tf session
    saver = tf.train.Saver()
    with tf.Session(config=config) as sess:
        # load from previous save
        if len(args.ckpt_name) > 0:
            saver.restore(sess, tf.train.latest_checkpoint('experiments/{}/{}/checkpoints/'.format(args.exp_name, args.exp_id)))
            print("Saver loaded.")

        # Test
        # Get inputs (test trajectory that is twice the size of a standard sequence)
        x = np.zeros((replay_memory.x_test.shape[0], args.trial_len, args.state_dim), dtype=np.float32)
        u = np.zeros((replay_memory.x_test.shape[0], args.trial_len - 1, args.action_dim), dtype=np.float32)
        x[:] = replay_memory.x_test[:]
        u[:] = replay_memory.u_test[:]

        # Find number of times to feed in input
        n_passes = (x.shape[0] // args.batch_size)

        # Initialize array to hold predictions
        preds = np.zeros((1, args.trial_len, args.state_dim))
        for t in range(n_passes):
            print("{}/{}".format(t, n_passes))
            tempx = x[args.batch_size * t : args.batch_size * (t + 1), :args.trial_len]
            tempu = u[args.batch_size * t : args.batch_size * (t + 1), :args.trial_len - 1]

            reshaped = tempx[:, :args.seq_length]

            # Construct inputs for network
            feed_in = {}
            feed_in[net.is_train] = False
            feed_in[net.feat_x] = np.reshape(reshaped, (args.batch_size * args.seq_length, args.state_dim))
            feed_in[net.x] = np.reshape(tempx, (args.batch_size * args.trial_len, args.state_dim))
            feed_in[net.u] = tempu
            feed_out = net.state_pred
            out = sess.run(feed_out, feed_in)
            x_pred = out.reshape(args.batch_size, args.trial_len, args.state_dim)

            # Append new set of predictions
            preds = np.vstack((preds, x_pred))

        preds = preds[1:]
        preds = np.reshape(preds, [preds.shape[0], args.trial_len, 1, int(np.sqrt(args.state_dim)), int(np.sqrt(args.state_dim))])

        # Save into mat file
        if not os.path.exists('experiments/{}/{}/{}'.format(args.exp_name, args.exp_id, args.exp_type)):
            os.mkdir('experiments/{}/{}/{}'.format(args.exp_name, args.exp_id, args.exp_type))
        np.save('experiments/{}/{}/{}/groundtruth.npy'.format(args.exp_name, args.exp_id, args.exp_type), x)
        np.save('experiments/{}/{}/{}/reconstructions.npy'.format(args.exp_name, args.exp_id, args.exp_type), preds)
        sio.savemat('experiments/{}/{}/{}/recons.mat'.format(args.exp_name, args.exp_id, args.exp_type), {'inputs': x, 'outputs': preds})
        return


if __name__ == '__main__':
    main()
