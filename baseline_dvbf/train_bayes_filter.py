import json
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import random
import argparse
import numpy as np
import tensorflow as tf

from tqdm import tqdm
from bayes_filter import BayesFilter
from replay_memory import BouncingBallData
from utils import visualize_bb, plot_metric



def main():
    ######################################
    #          General Params            #
    ######################################
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', type=str, default='bf_checkpoints',
                        help='directory to store checkpointed models')
    parser.add_argument('--val_frac', type=float, default=0.1, help='fraction of data to be witheld in validation set')
    parser.add_argument('--ckpt_name', type=str, default='', help='name of checkpoint file to load (blank means none)')
    parser.add_argument('--save_name', type=str, default='bf_model', help='name of checkpoint files for saving')
    parser.add_argument('--domain_name', type=str, default='Pendulum-v0', help='environment name')
    parser.add_argument('--device', type=str, default='0', help='which GPU to use')

    parser.add_argument('--exp_name', type=str, default='grav16', help='name of the experiment to save')
    parser.add_argument('--exp_id', type=str, default='1', help='id of the experiment')
    parser.add_argument('--exp_type', type=str, default='mixed_gravity_16', help='type of the experiment, ie mixed')

    parser.add_argument('--dataset_path', type=str, default='mixed_inc_data/mixed15/', help='path for the dataset used')
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
    parser.add_argument('--trial_len', type=int, default=25, help='number of steps in each trial')
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
    args.state_dim = args.state_dim
    args.action_dim = 1

    # Set random seed
    random.seed(1)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    # Specify experiment folder
    if not os.path.exists('experiments/'):
        os.makedirs('experiments/')

    if not os.path.exists('experiments/{}/'.format(args.exp_name)):
        os.makedirs('experiments/{}/'.format(args.exp_name))

    if not os.path.exists('experiments/{}/{}/'.format(args.exp_name, args.exp_id)):
        os.makedirs('experiments/{}/{}/'.format(args.exp_name, args.exp_id))

    if not os.path.exists('experiments/{}/{}/reconstructions/'.format(args.exp_name, args.exp_id)):
        os.makedirs('experiments/{}/{}/reconstructions/'.format(args.exp_name, args.exp_id))

    if not os.path.exists('experiments/{}/{}/checkpoints/'.format(args.exp_name, args.exp_id)):
        os.makedirs('experiments/{}/{}/checkpoints/'.format(args.exp_name, args.exp_id))

    with open('experiments/{}/{}/args.txt'.format(args.exp_name, args.exp_id), 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    # Construct model
    net = BayesFilter(args)
    train(args, net)


# Train network
def train(args, net):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    # Begin tf session
    with tf.Session(config=config) as sess:
        # Initialize variables
        tf.global_variables_initializer().run()
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)

        # load from previous save
        if len(args.ckpt_name) > 0:
            saver.restore(sess, os.path.join(args.save_dir, args.ckpt_name))

        # Generate the dataset
        replay_memory = BouncingBallData(args, file_path=args.dataset_path + args.exp_type)

        # Function to evaluate loss on validation set
        def val_loss(kl_weight):
            args.test_set = True
            replay_memory.reset_batchptr_val()
            loss = 0.0
            for b in range(replay_memory.n_batches_val):
                # Get inputs
                batch_dict = replay_memory.next_batch_val()
                x = batch_dict["states"]
                u = batch_dict['inputs']

                # Construct inputs for network
                feed_in = {}
                feed_in[net.is_train] = False
                feed_in[net.x] = np.reshape(x, (args.batch_size * args.seq_length, args.state_dim))
                feed_in[net.u] = u
                if args.kl_weight > 0.0:
                    feed_in[net.kl_weight] = kl_weight
                else:
                    feed_in[net.kl_weight] = 1.0

                # Find loss
                feed_out = net.cost
                cost = sess.run(feed_out, feed_in)
                loss += cost

            return loss / replay_memory.n_batches_val

        # Initialize variable to track validation score over time
        count_decay = 0

        # Define temperature for annealing kl_weight
        T = args.anneal_time * replay_memory.n_batches_train
        count = 0

        # Arrays to hold epoch metrics
        tr_losses, val_losses = [], []
        tr_rec, val_rec = [], []
        tr_kl, val_kl = [], []

        # Loop over epochs
        for e in range(args.num_epochs):
            visualize_bb(e, args, sess, net, replay_memory)

            # Initialize loss
            loss = 0.0
            rec_loss = 0.0
            kl_loss = 0.0
            loss_count = 0
            replay_memory.reset_batchptr_train()

            # Loop over batches
            for b in tqdm(range(replay_memory.n_batches_train)):
                start = time.time()
                count += 1

                # Update kl_weight
                if e < args.start_kl:
                    kl_weight = 1e-3
                else:
                    count += 1
                    kl_weight = min(args.kl_weight, 1e-3 + args.kl_weight * count / float(T))

                # Get inputs
                batch_dict = replay_memory.next_batch_train()
                x = batch_dict["states"]
                u = batch_dict['inputs']

                # Construct inputs for network
                feed_in = {}
                feed_in[net.is_train] = True
                feed_in[net.x] = np.reshape(x, (args.batch_size * args.seq_length, args.state_dim))
                feed_in[net.feat_x] = np.reshape(x[:, :args.seq_length], (args.batch_size * args.seq_length, args.state_dim))
                feed_in[net.u] = u
                feed_in[net.kl_weight] = kl_weight

                # Find loss and perform training operation
                feed_out = [net.cost, net.loss_reconstruction, net.kl_loss, net.train]
                out = sess.run(feed_out, feed_in)

                # Update and display cumulative losses
                loss += out[0]
                rec_loss += out[1]
                kl_loss += out[2]
                loss_count += 1

                end = time.time()

            print("KL Weights: ", kl_weight)

            # Print train epoch
            print("(epoch {}), train_loss = {:.3f}, time/batch = {:.3f}".format(e, loss / loss_count, end - start))
            print("(epoch {}), rec_loss = {:.3f}, time/batch = {:.3f}".format(e, rec_loss / loss_count, end - start))
            print("(epoch {}), kl_loss = {:.3f}, time/batch = {:.3f}".format(e, kl_loss / loss_count, end - start))

            # Evaluate loss on validation set
            # score = val_loss(args.kl_weight * (e >= args.start_kl))
            score = 0
            print('Validation Loss: {0:f}'.format(score))

            # Set learning rate
            if e % 2 == 0:
                count_decay += 1
                print('setting learning rate to ', args.learning_rate * (args.decay_rate ** count_decay))
                sess.run(tf.assign(net.learning_rate, args.learning_rate * (args.decay_rate ** count_decay)))

            # Save epoch metrics
            tr_losses.append(loss / loss_count)
            val_losses.append(score)
            tr_rec.append(rec_loss / loss_count)
            tr_kl.append(kl_loss / loss_count)

            # Plot them
            plot_metric(args.exp_name, args.exp_id, 'losses', (tr_losses, val_losses), 'Losses', mse=True)
            plot_metric(args.exp_name, args.exp_id, 'tr_rec', tr_rec, 'Reconstruction likelihood')
            plot_metric(args.exp_name, args.exp_id, 'tr_kl', tr_rec, 'KL Beta')

            # Save model every epoch
            saver.save(sess, 'experiments/{}/{}/checkpoints/ep{}.ckpt'.format(args.exp_name, args.exp_id, e), global_step=e)
            print("model saved to {}".format('experiments/{}/{}/checkpoints/'.format(args.exp_name, args.exp_id)))


if __name__ == '__main__':
    main()
