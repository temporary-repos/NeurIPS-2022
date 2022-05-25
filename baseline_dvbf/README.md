# vbf
Implementation of Deep Variational Bayes Filter, modified from the codebase: https://github.com/Jgmorton/vbf.
It has been modified to allow for variable training and generation lengths, perform initial state generation with an encoder,
and to have visualization/outputs in a consistent format.

The data loading has been modified to simply require the data format of the proposed model and is easy to load in.
Note that the training start-up for this model takes a significant amount of time and it requires a very specific 
anaconda environment to function. Included in the repo is the requirements folder to setup that env.

Example command:

`python3 train_bayes_filter.py --seq_length 32 --extractor_size 64 64 --inference_size 64 64  --kl_weight 0.1`