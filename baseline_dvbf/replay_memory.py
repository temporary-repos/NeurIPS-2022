import math
import numpy as np


# Class to load and preprocess data
class BouncingBallData():
    def __init__(self, args, file_path):
        # Base lines
        self.batch_size = args.batch_size
        self.seq_length = args.seq_length

        # Load train/test npz files
        if args.test == 1:
            npzfile = np.load(file_path + '.npz')
            npztestfile = np.load(file_path + '.npz')
        else:
            npzfile = np.load(file_path + '_train.npz')
            npztestfile = np.load(file_path + '_valid.npz')

        # Load data
        self.train_images = npzfile['images'].astype(np.float32)[:, :self.seq_length]
        self.train_images = (self.train_images > 0).astype('float32')
        self.x = self.train_images.reshape(self.train_images.shape[0], self.seq_length, -1)
        self.u = np.zeros((self.train_images.shape[0],  self.seq_length - 1, 1), dtype=np.float32)

        # Get data dimensions
        self.sequences, self.timesteps, self.d1 = self.x.shape

        # Test data arrays
        self.x_test = npztestfile['images'].astype(np.float32)[:, :args.trial_len]
        self.x_test = (self.x_test > 0).astype('float32')
        self.x_test = self.x_test.reshape(self.x_test.shape[0], args.trial_len, -1)
        self.u_test = np.zeros((self.x_test.shape[0], args.trial_len - 1, 1), dtype=np.float32)

        # Split into train/val
        self._create_inputs_targets(args)
        self._create_split(args)

    def _create_inputs_targets(self, args):
        # Create batch_dict
        self.batch_dict = {}

        # Print tensor shapes
        print('states: ', self.x.shape)
        print('inputs: ', self.u.shape)

        self.batch_dict['states'] = np.zeros((args.batch_size, self.seq_length, args.state_dim))
        self.batch_dict['inputs'] = np.zeros((args.batch_size, self.seq_length - 1, args.action_dim))

        # Shuffle data before splitting into train/val
        print('shuffling...')
        p = np.random.permutation(len(self.x))
        self.x = self.x[p]
        self.u = self.u[p]

    # Separate data into train/validation sets
    def _create_split(self, args):
        # Compute number of batches
        self.n_batches = len(self.x) // args.batch_size
        self.n_batches_val = int(math.floor(args.val_frac * self.n_batches))
        self.n_batches_train = self.n_batches - self.n_batches_val

        print('num training batches: ', self.n_batches_train)
        print('num validation batches: ', self.n_batches_val)

        # Divide into train and validation datasets
        self.x_val = self.x[self.n_batches_train * args.batch_size:]
        self.u_val = self.u[self.n_batches_train * args.batch_size:]
        self.x = self.x[:self.n_batches_train * args.batch_size]
        self.u = self.u[:self.n_batches_train * args.batch_size]

        # Set batch pointer for training and validation sets
        self.reset_batchptr_train()
        self.reset_batchptr_val()

    # Sample a new batch of data
    def next_batch_train(self):
        # Extract next batch
        batch_index = self.batch_permuation_train[
                      self.batchptr_train * self.batch_size:(self.batchptr_train + 1) * self.batch_size]
        self.batch_dict['states'] = self.x[batch_index]
        self.batch_dict['inputs'] = self.u[batch_index]

        # Update pointer
        self.batchptr_train += 1
        return self.batch_dict

    # Return to first batch in train set
    def reset_batchptr_train(self):
        self.batch_permuation_train = np.random.permutation(len(self.x))
        self.batchptr_train = 0

    # Return next batch of data in validation set
    def next_batch_val(self):
        # Extract next validation batch
        batch_index = range(self.batchptr_val * self.batch_size, (self.batchptr_val + 1) * self.batch_size)
        self.batch_dict['states'] = self.x_val[batch_index]
        self.batch_dict['inputs'] = self.u_val[batch_index]

        # Update pointer
        self.batchptr_val += 1
        return self.batch_dict

    # Return to first batch in validation set
    def reset_batchptr_val(self):
        self.batchptr_val = 0

