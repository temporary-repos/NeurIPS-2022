import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def visualize_bb(ep, args, sess, net, replay_memory):
    # Get inputs (test trajectory that is twice the size of a standard sequence)
    x = np.zeros((args.batch_size, args.seq_length, args.state_dim), dtype=np.float32)
    u = np.zeros((args.batch_size, args.seq_length - 1, args.action_dim), dtype=np.float32)
    x[:] = replay_memory.x[:args.batch_size]
    u[:] = replay_memory.u[:args.batch_size]

    # Construct inputs for network
    feed_in = {}
    feed_in[net.is_train] = True
    feed_in[net.x] = np.reshape(x, (args.batch_size * args.seq_length, args.state_dim))
    feed_in[net.feat_x] = np.reshape(x[:, :args.seq_length], (args.batch_size * args.seq_length, args.state_dim))
    feed_in[net.u] = u

    feed_out = net.state_pred
    out = sess.run(feed_out, feed_in)

    x = x.reshape(args.batch_size, args.seq_length, 1, 32, 32)
    x_pred = out.reshape(args.batch_size, args.seq_length, 1, 32, 32)
    plot_recon(x, x_pred, args.seq_length, "", "",
               "experiments/{}/{}/reconstructions/recon_train{}".format(args.exp_name, args.exp_id, ep))

    # Get inputs (test trajectory that is twice the size of a standard sequence)
    x = np.zeros((args.batch_size, args.trial_len, args.state_dim), dtype=np.float32)
    u = np.zeros((args.batch_size, args.trial_len - 1, args.action_dim), dtype=np.float32)
    x[:] = replay_memory.x_test[:args.batch_size]
    u[:] = replay_memory.u_test[:args.batch_size]

    # Construct inputs for network
    feed_in = {}
    feed_in[net.is_train] = False
    feed_in[net.x] = np.reshape(x, (args.batch_size * args.trial_len, args.state_dim))
    feed_in[net.feat_x] = np.reshape(x[:, :args.seq_length], (args.batch_size * args.seq_length, args.state_dim))
    feed_in[net.u] = u

    feed_out = net.state_pred
    out = sess.run(feed_out, feed_in)

    x = x.reshape(args.batch_size, args.trial_len, 1, 32, 32)
    x_pred = out.reshape(args.batch_size, args.trial_len, 1, 32, 32)
    plot_recon(x, x_pred, args.seq_length, "", "",
               "experiments/{}/{}/reconstructions/recon_test{}".format(args.exp_name, args.exp_id, ep))


def visualize_rot(ep, args, sess, net, replay_memory):
    # Get inputs (test trajectory that is twice the size of a standard sequence)
    x = np.zeros((args.batch_size, args.seq_length, args.state_dim), dtype=np.float32)
    u = np.zeros((args.batch_size, args.seq_length - 1, args.action_dim), dtype=np.float32)
    x[:] = replay_memory.x[:args.batch_size]
    u[:] = replay_memory.u[:args.batch_size]

    # Construct inputs for network
    feed_in = {}
    feed_in[net.is_train] = True
    feed_in[net.x] = np.reshape(x, (args.batch_size * args.seq_length, args.state_dim))
    feed_in[net.feat_x] = np.reshape(x[:, :args.seq_length], (args.batch_size * args.seq_length, args.state_dim))
    feed_in[net.u] = u

    feed_out = net.state_pred
    out = sess.run(feed_out, feed_in)

    x = x.reshape(args.batch_size, args.seq_length, 1, 32, 32)
    x_pred = out.reshape(args.batch_size, args.seq_length, 1, 32, 32)
    plot_rot_recon(x, x_pred, args.seq_length, "", "",
               "experiments/{}/{}/reconstructions/recon_train{}".format(args.exp_name, args.exp_id, ep))

    # Get inputs (test trajectory that is twice the size of a standard sequence)
    x = np.zeros((args.batch_size, args.trial_len, args.state_dim), dtype=np.float32)
    u = np.zeros((args.batch_size, args.trial_len - 1, args.action_dim), dtype=np.float32)
    x[:] = replay_memory.x_test[:args.batch_size]
    u[:] = replay_memory.u_test[:args.batch_size]

    # Construct inputs for network
    feed_in = {}
    feed_in[net.is_train] = False
    feed_in[net.x] = np.reshape(x, (args.batch_size * args.trial_len, args.state_dim))
    feed_in[net.feat_x] = np.reshape(x[:, :args.seq_length], (args.batch_size * args.seq_length, args.state_dim))
    feed_in[net.u] = u

    feed_out = net.state_pred
    out = sess.run(feed_out, feed_in)

    x = x.reshape(args.batch_size, args.trial_len, 1, 32, 32)
    x_pred = out.reshape(args.batch_size, args.trial_len, 1, 32, 32)
    plot_rot_recon(x, x_pred, args.seq_length, "", "",
               "experiments/{}/{}/reconstructions/recon_test{}".format(args.exp_name, args.exp_id, ep))


def visualize_predictions(args, sess, net, replay_memory, e=0):
    # Get inputs (test trajectory that is twice the size of a standard sequence)
    x = np.zeros((args.batch_size, 2 * args.seq_length, args.state_dim), dtype=np.float32)
    u = np.zeros((args.batch_size, 2 * args.seq_length - 1, args.action_dim), dtype=np.float32)
    x[:] = replay_memory.x_test
    u[:] = replay_memory.u_test

    # Find number of times to feed in input
    n_passes = 200 // args.batch_size

    # Initialize array to hold predictions
    preds = np.zeros((1, 2 * args.seq_length, args.state_dim))
    for t in range(n_passes):
        # Construct inputs for network
        feed_in = {}
        feed_in[net.x] = np.reshape(x, (2 * args.batch_size * args.seq_length, args.state_dim))
        feed_in[net.u] = u
        feed_out = net.state_pred
        out = sess.run(feed_out, feed_in)
        x_pred = out.reshape(args.batch_size, 2 * args.seq_length, args.state_dim)

        # Append new set of predictions
        preds = np.concatenate((preds, x_pred), axis=0)
    preds = preds[1:]

    # Find mean, max, and min of predictions
    pred_mean = np.mean(preds, axis=0)
    pred_std = np.std(preds, axis=0)
    pred_min = np.amin(preds, axis=0)
    pred_max = np.amax(preds, axis=0)

    diffs = np.linalg.norm(
        (preds[:, :args.seq_length] - sess.run(net.shift)) / sess.run(net.scale) - x[0, :args.seq_length], axis=(1, 2))
    best_pred = np.argmin(diffs)
    worst_pred = np.argmax(diffs)

    # Plot different quantities
    x = x * sess.run(net.scale) + sess.run(net.shift)

    # Find indices for random predicted trajectories to plot
    ind0 = best_pred
    ind1 = worst_pred

    # Plot values
    plt.close()
    f, axs = plt.subplots(args.state_dim, sharex=True, figsize=(15, 15))
    plt.rc('text', usetex=False)
    plt.rc('font', family='serif')

    for i in range(args.state_dim):
        axs[i].plot(range(2 * args.seq_length), x[0, :, i], 'k')
        axs[i].plot(range(2 * args.seq_length), preds[ind0, :, i], 'r')
        axs[i].plot(range(2 * args.seq_length), preds[ind1, :, i], 'g')
        axs[i].plot(range(2 * args.seq_length), pred_mean[:, i], 'b')
        axs[i].fill_between(range(2 * args.seq_length), pred_min[:, i], pred_max[:, i], facecolor='blue', alpha=0.5)
        axs[i].set_ylim([np.amin(x[0, :, i]) - 0.2, np.amax(x[0, :, i]) + 0.2])

    plt.xlabel('Time Step')
    plt.xlim([0, 2 * args.seq_length])
    plt.savefig('bf_predictions/predictions_' + str(e) + '.png')


def plot_recon(X, Xrec, genstart, exp_name, exp_id, fname):
    [num_sample, time_steps, _, _, _] = X.shape
    blank = 5

    panel = np.ones((32 * 2 * num_sample + blank * (num_sample + 2), 32 * time_steps + 2 * blank)) * 255
    panel = np.uint8(panel)
    panel = Image.fromarray(panel)

    selected_idx = np.random.choice(X.shape[0], num_sample, replace=False)
    selected_idx = sorted(selected_idx)

    for num, idx in enumerate(selected_idx):
        selected_inps = X[idx]
        selected_rcns = Xrec[idx, :genstart]
        selected_gens = Xrec[idx, genstart:]

        selected_inps = np.uint8(selected_inps * 255)
        selected_rcns = np.uint8(selected_rcns * 255)
        selected_gens = np.uint8(selected_gens * 255)

        img = np.zeros((32 * 2, genstart * 32)).astype(np.uint8)
        for i in range(genstart):
            img[:32, i * 32: (i + 1) * 32] = selected_inps[i]
            img[32:64, i * 32: (i + 1) * 32] = selected_rcns[i]

        img = Image.fromarray(img)
        panel.paste(img, (blank, blank * (num + 1) + num * 32 * 2))

        img_gen = np.zeros((32 * 2, (time_steps - genstart) * 32)).astype(np.uint8)
        for i in range(time_steps - genstart):
            img_gen[:32, i * 32: (i + 1) * 32] = selected_inps[i + genstart]
            img_gen[32:64, i * 32: (i + 1) * 32] = selected_gens[i]

        img_gen = Image.fromarray(img_gen)
        panel.paste(img_gen, (blank * 2 + 32 * genstart, blank * (num + 1) + num * 32 * 2))

    panel.save('{}.png'.format(fname))


def plot_rot_recon(X, Xrec, genstart, exp_name, exp_id, fname):
    [num_sample, time_steps, _, _, _] = X.shape
    blank = 5

    panel = np.ones((32 * 2 * num_sample + blank * (num_sample + 2), 32 * time_steps + 2 * blank)) * 255
    panel = np.uint8(panel)
    panel = Image.fromarray(panel)

    selected_idx = np.random.choice(X.shape[0], num_sample, replace=False)
    selected_idx = sorted(selected_idx)

    for num, idx in enumerate(selected_idx):
        selected_inps = X[idx]
        selected_rcns = Xrec[idx, :genstart]
        selected_gens = Xrec[idx, genstart:]

        selected_inps = np.uint8(selected_inps * 255)
        selected_rcns = np.uint8(selected_rcns * 255)
        selected_gens = np.uint8(selected_gens * 255)

        img = np.zeros((32 * 2, genstart * 32)).astype(np.uint8)
        for i in range(genstart):
            img[:32, i * 32: (i + 1) * 32] = selected_inps[i]
            img[32:56, i * 32: (i + 1) * 32] = selected_rcns[i]

        img = Image.fromarray(img)
        panel.paste(img, (blank, blank * (num + 1) + num * 32 * 2))

        img_gen = np.zeros((32 * 2, (time_steps - genstart) * 32)).astype(np.uint8)
        for i in range(time_steps - genstart):
            img_gen[:32, i * 32: (i + 1) * 32] = selected_inps[i + genstart]
            img_gen[32:56, i * 32: (i + 1) * 32] = selected_gens[i]

        img_gen = Image.fromarray(img_gen)
        panel.paste(img_gen, (blank * 2 + 32 * genstart, blank * (num + 1) + num * 32 * 2))

    panel.save('{}.png'.format(fname))


def plot_metric(enum, eid, metric, values, title, mse=False):
    """ Plotting function for metrics """
    plt.figure(2)
    plt.title(title)
    plt.xlabel("Epoch")

    if mse is True:
        plt.plot(values[0])
        plt.plot(values[1])
        plt.legend(('train', 'test'))
    else:
        plt.plot(values)
    plt.savefig('experiments/{}/{}/{}.png'.format(enum, eid, metric))
    plt.close()