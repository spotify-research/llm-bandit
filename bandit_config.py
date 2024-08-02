from enum import Enum

import numpy as np

class BanditConfig(Enum):

    # Thompson Sampling or Greedy
    IS_TS = False

    INIT_FROM = 'gpt2'

    NO_OUTER_LOOPS = 100
    '''
    Notice: it is equivalent to change no_inner_loops or samples_per_inner_loop in
    our script. By changing samples_per_inner_loop we have a slightly better usage of the 
    VRAM, but it is almost negligible.
    '''
    NO_INNER_LOOPS = 1
    SAMPLES_PER_INNER_LOOP = 32

    N_EPOCHS = 50

    # learning rate suggested by Karpathy in NanoGPT for fine-tuning GPT2 on Shakespeare
    LEARNING_RATE = 3e-5

    OBSERVATION_VARIANCE = 0.01
    PRIOR_VARIANCE = 0.01

    '''
    If fixed_num_iterations is True, we train exactly for 'n_epochs' iterations, 
    no matter how much data we have. 
    Else, we try to see each data point for 'n_epochs' times.
    '''
    FIXED_NUM_ITERATIONS = True

    # If USE_ADAM_W==False, we use standard Adam.
    # AdamW was suggested by Karpathy.
    USE_ADAM_W = False

    IS_DECAYED_LR = False  # if we want to use lr decay - Karpathy suggests not to use it for fine-tuning
    WARMUP_ITERS = 100  # how many steps to warm up for
    LR_DECAY_ITERS = 9600  # should be ~= max_iters per Chinchilla
    MIN_LR = LEARNING_RATE / 10  # minimum learning rate, should be ~= learning_rate/10 per Chinchilla

    '''
    We can also set a maximum number of iterations.
    Let's say that we have n_epochs=50 and n_data=100*loader_batch_size.
    If we don't set a max_iter, we will have to do 50*100=5000 iterations of training.
    If we set max_iter = 3000, we will try to see each data point n_epochs times until
    we reach 3000 iterations, then we cap the number of iterations. 
    '''
    MAX_ITER = np.inf

    '''
    If incremental_training_enabled=False, we create a Bandit initialized to GPT2 pre-trained
    for each round. Otherwise, we keep the bandit trained at the previous round for the next round.
    '''
    INCREMENTAL_TRAINING_ENABLED = True

    HESSIAN_SUBSAMPLE = 32  # how many data points to use to compute the Hessian
