#!/usr/bin/env python
# coding: utf-8

import gc
import random
import torch
import numpy as np
import pytorch_lightning as pl
import os
import copy

from utils.data_utils import generate_random_dataset, \
    generate_dataset_sampling, generate_dataset_sampling_last_layer, generate_dataset_sampling_dropout, \
    get_train_loader_from_dataset, CustomTrainingDataset, generate_dataset_greedy, dtype, ptdtype, \
    generate_dataset_sampling_last_layer_and_diag

from bandit_config import BanditConfig

import argparse

from PretrainedBanditModelHF import PretrainedBanditModelHF, NewPretrainedBanditModelEpiNetHF, Epinet5

# Set up the argparse parser
from utils.training_utils import get_lr, get_hessian
from utils import data_utils
from utils.data_utils import save_regret_trace

parser = argparse.ArgumentParser(description='Train a Multi-Armed Bandit model.')
parser.add_argument('--subsample', type=int, default=None, help='Subsample parameter for training.')
parser.add_argument('--n_epochs', type=int, default=None, help='N. epochs for training.')
parser.add_argument('--seed', type=int, default=None, help='Random seed.')
parser.add_argument('--prior_variance', type=float, default=None, help='Prior variance')
parser.add_argument('--obs_var_ratio', type=float, default=None, help='Obs to prior variance ratio')
parser.add_argument('--obs_var', type=float, default=None, help='Observation variance')
parser.add_argument('--ts', choices=['last_la', 'la', 'epinet', 'dropout'],
                    help='Type of Thompson sampling variant')

args = parser.parse_args()

assert args.obs_var_ratio is None or args.obs_var is None, "provided both obs. var ratio and obs. variance."

# -----------------------------------------------------------------------------
seed = 0
if args.seed is not None:
    seed += args.seed

device = 'cuda'
compile = False  # use PyTorch 2.0 to compile the model to be faster
# exec(open('configurator.py').read()) # overrides from command line or config file
# -----------------------------------------------------------------------------

'''
--------------------------------------------------------------------------------------------------------------
--------------------------------------------------------------------------------------------------------------
------------------------------------------------GLOBAL PROPERTIES---------------------------------------------
--------------------------------------------------------------------------------------------------------------
--------------------------------------------------------------------------------------------------------------
'''

random.seed(seed)
np.random.seed(seed)

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu'  # for later use in torch.autocast
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

pl.seed_everything(seed)

# block_size is the number of tokens for each data point
block_size = data_utils.block_size

# loader_batch_size: batch size of the data loader
loader_batch_size = data_utils.loader_batch_size
gradient_accumulation_steps = data_utils.gradient_accumulation_steps

# adam optimizer params suggested by Karpathy
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0  # clip gradients at this value, or disable if == 0.0

loss_scale_factor = 100.0


DEBUG = False


print_action_selection = False

is_ts = bool(BanditConfig.IS_TS.value)

init_from = BanditConfig.INIT_FROM.value

repetitions = 1
no_outer_loops = BanditConfig.NO_OUTER_LOOPS.value
no_inner_loops = BanditConfig.NO_INNER_LOOPS.value
samples_per_inner_loop = BanditConfig.SAMPLES_PER_INNER_LOOP.value
n_epochs = BanditConfig.N_EPOCHS.value
is_decayed_lr = BanditConfig.IS_DECAYED_LR.value
learning_rate = BanditConfig.LEARNING_RATE.value
observation_variance = BanditConfig.OBSERVATION_VARIANCE.value
prior_variance = BanditConfig.PRIOR_VARIANCE.value

fixed_num_iterations = BanditConfig.FIXED_NUM_ITERATIONS.value
use_adam_w = BanditConfig.USE_ADAM_W.value
warmup_iters = BanditConfig.WARMUP_ITERS.value
lr_decay_iters = BanditConfig.LR_DECAY_ITERS.value
min_lr = BanditConfig.MIN_LR.value
max_iter = BanditConfig.MAX_ITER.value
h_subsample = BanditConfig.HESSIAN_SUBSAMPLE.value

incremental_training_enabled = bool(BanditConfig.INCREMENTAL_TRAINING_ENABLED.value)


config_dict = {name: elem.value for name, elem in BanditConfig.__members__.items()}

config_dict['SEED'] = seed

if args.subsample is not None:
    h_subsample = args.subsample
    config_dict['HESSIAN_SUBSAMPLE'] = h_subsample
    print(f"Overwriting hessian subsample with value: {h_subsample}")
else:
    print(f"Using default hessian subsample: {h_subsample}")

if args.n_epochs is not None:
    n_epochs = args.n_epochs
    config_dict['N_EPOCHS'] = n_epochs
    print(f"Overwriting n_epochs with value: {n_epochs}")
else:
    print(f"Using default n_epochs: {n_epochs}")

if args.prior_variance is not None:
    prior_variance = args.prior_variance
    config_dict['PRIOR_VARIANCE'] = prior_variance
    print(f"Overwriting prior_variance with value: {prior_variance}")
else:
    print(f"Using default prior_variance: {prior_variance}")

if args.obs_var_ratio is not None:
    observation_variance = args.obs_var_ratio * prior_variance
    config_dict['OBSERVATION_VARIANCE'] = observation_variance
    print(f"Overwriting observation_variance with value: {observation_variance}")
else:
    print(f"Using default observation_variance: {observation_variance}")

if args.obs_var is not None:
    observation_variance = args.obs_var
    config_dict['OBSERVATION_VARIANCE'] = observation_variance
    print(f"Overwriting observation_variance with value: {observation_variance}")
else:
    print(f"Using default observation_variance: {observation_variance}")


if args.seed is not None:
    config_dict['SEED'] = seed
    print(f"Overwriting random seed with value: {seed}")
else:
    print(f"Using default random seed: {seed}")

from datetime import datetime

log_file = open("main.log", "a")


log_file.write('\n')
log_file.write(str(datetime.now()))
log_file.write('\n')
# print global properties
for name, value in config_dict.items():
    log_file.write(f'\n{name} = {value}')


'''
Here we get the pre-trained GPT2 model.
'''

# create the bandit model with pre-trained GPT2 skeleton
bandit_model = PretrainedBanditModelHF(init_from)
bandit_model.to(device, dtype=ptdtype)

# compile the model
if compile:
    print("compiling the model... (takes a ~minute)")
    # unoptimized_model = bandit_model
    bandit_model = torch.compile(bandit_model)  # requires PyTorch 2.0

print(bandit_model)

# Helper function that saves the regret trace to disk.
def generate_folder_name(dir_dict):
    if not dir_dict['IS_DECAYED_LR']:
        del dir_dict['WARMUP_ITERS']
        del dir_dict['LR_DECAY_ITERS']
        del dir_dict['MIN_LR']

    if not dir_dict['IS_TS']:
        del dir_dict['HESSIAN_SUBSAMPLE']

    if 'imdb' in data_utils.data_dir:
        folder_name = 'IMDB/'
    elif 'toxic' in data_utils.data_dir:
        folder_name = f'TOXIC/'
    elif 'tweet_eval/hate' in data_utils.data_dir:
        folder_name = f'TWEET_HATE'
    elif 'tweet_eval/offensive' in data_utils.data_dir:
        folder_name = f'TWEET_OFFENSIVE'
    else:
        raise ValueError(data_utils.data_dir)

    for k, v in dir_dict.items():
        folder_name = os.path.join(folder_name, f'{k}={str(v)}')

    return folder_name


from torch.utils.data import DataLoader

import time

# 'set_detect_anomaly' is needed to detect NaN gradients:
# it raises an error whenever gradients are NaN.
torch.autograd.set_detect_anomaly(True)

# If the model can't be compiled (PyTorch 2), "suppress_errors" needed to
# fall back to not compiled model without crashing
import torch._dynamo

torch._dynamo.config.suppress_errors = True


def get_trained_model(model,
                      train_loader,
                      n_epochs,
                      scheduler_it,
                      observation_variance,
                      prior_precision=1/prior_variance,
                      prior_weights=None,
                      anneal_regul=False,
                      ):
    criterion = torch.nn.MSELoss()
    if anneal_regul:
        curr_data_len = len(D)
    else:
        curr_data_len = len(train_loader.dataset)
    # if use_adam_w:
    #    weight_decay = (observation_variance * prior_precision) / (
    #            len(D) * 2.0  * loss_scale_factor)
    #    optimizer = torch.optim.AdamW(model.parameters(), lr=get_lr(scheduler_it),
    #                                  eps=1e-6,
    #                                  weight_decay=weight_decay,
    #                                  betas=(beta1, beta2)
    #                                  )
    # else:
    if isinstance(prior_precision, float) and prior_weights is None:
        weight_decay = (observation_variance * prior_precision) / (
                curr_data_len * 2.0)
    else:
        weight_decay = 0.0

    current_lr = get_lr(scheduler_it,
                        is_decayed_lr=is_decayed_lr,
                        warmup_iters=warmup_iters,
                        learning_rate=learning_rate,
                        lr_decay_iters=lr_decay_iters,
                        min_lr=min_lr,
                        )
    optimizer = torch.optim.Adam(model.parameters(), current_lr,
                                 eps=1e-6,
                                 betas=(beta1, beta2),
                                 weight_decay=weight_decay,
                                 )

    # Creates a GradScaler once at the beginning of training.
    scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

    i = 0
    start_time = time.time()

    n_batches = max(curr_data_len // loader_batch_size, 1)

    if fixed_num_iterations:
        train_limit = n_epochs
    else:
        train_limit = min(n_epochs * n_batches, max_iter)

    while True:
        debug_likelihood = 0.0
        i += 1

        optimizer.zero_grad()

        current_lr = get_lr(scheduler_it,
                            is_decayed_lr=is_decayed_lr,
                            warmup_iters=warmup_iters,
                            learning_rate=learning_rate,
                            lr_decay_iters=lr_decay_iters,
                            min_lr=min_lr,
                            )

        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr

        batch_contexts, batch_attention_mask, batch_actions, batch_rewards = next(iter(train_loader))
        pointer = 0
        micro_batch = loader_batch_size // gradient_accumulation_steps
        for micro_step in range(gradient_accumulation_steps):
            contexts = batch_contexts[pointer:pointer+micro_batch]
            actions = batch_actions[pointer:pointer+micro_batch]
            attention_mask = batch_attention_mask[pointer:pointer+micro_batch]
            rewards = batch_rewards[pointer:pointer+micro_batch]

            contexts = contexts.cuda()
            actions = actions.cuda()
            attention_mask = attention_mask.cuda()
            rewards = rewards.cuda()

            likelihood_term = criterion(torch.gather(model(contexts, attention_mask=attention_mask),
                                                     dim=1, index=actions.reshape((-1, 1))),
                                        rewards.unsqueeze(dim=1)) / (
                                  2.0)

            loss_val = (likelihood_term) / (loss_scale_factor * gradient_accumulation_steps)
            try:
                # loss_val.backward()
                scaler.scale(loss_val).backward()
            except RuntimeError as err:
                print(f'Handling {err}')
                torch.save(model, 'bugged_model.pt')
                torch.save(contexts, 'contexts.pt')
                torch.save(actions, 'actions.pt')
                torch.save(rewards, 'rewards.pt')
                torch.save(optimizer.state_dict(), 'opt_state_dict.pt')

                raise err

            pointer += micro_batch


        if isinstance(prior_precision, float):
            if prior_weights is not None:

                coefficient = (observation_variance * prior_precision) / (curr_data_len * 2.0)
                regularization_term = 0.0
                pointer = 0
                for name, params in model.named_parameters():
                    curr_num_param = params.numel()
                    if name == 'last_layer.weight':
                        regularization_term += torch.sum(
                            params.view(-1) ** 2)
                    elif name != 'last_layer.bias':
                        # if 'bias' not in name:
                        regularization_term += (
                                (params.view(-1) - prior_weights[pointer:pointer + curr_num_param]) ** 2
                        ).sum()
                    pointer += curr_num_param

                regularization_term *= coefficient
            else:
                # This is 0.0 because the weight decay is already inside Adam
                regularization_term = 0.0
        else:

            regularization_term = 0.0
            pointer = 0
            for name, params in model.named_parameters():
                curr_num_param = params.numel()
                regularization_term += torch.dot(
                    torch.sub(params.view(-1),
                              prior_weights[pointer:pointer + curr_num_param]) ** 2,
                    prior_precision[pointer:pointer + curr_num_param]
                )

                pointer += curr_num_param

        loss_val = (regularization_term) / (loss_scale_factor)

        try:
            # loss_val.backward()
            scaler.scale(loss_val).backward()
        except RuntimeError as err:
            print(f'Handling {err}')
            torch.save(model, 'bugged_model.pt')
            torch.save(contexts, 'contexts.pt')
            torch.save(actions, 'actions.pt')
            torch.save(rewards, 'rewards.pt')
            torch.save(optimizer.state_dict(), 'opt_state_dict.pt')

            raise err

        # clip the gradient
        if grad_clip != 0.0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        # scaler.step() first unscales the gradients of the optimizer's assigned params.
        # If these gradients do not contain infs or NaNs, optimizer.step() is then called,
        # otherwise, optimizer.step() is skipped.
        scaler.step(optimizer)

        # Updates the scale for next iteration.
        scaler.update()

        if i % 5000 == 0 or i == 1:
            # print(f'Iteration {i}. Avg grad norm: {total_grad_norm / counter}')
            print(f'Iteration {i}.')
            print(f'Time passed: {(time.time() - start_time) / 60:.2f} min.')

        if i == train_limit:
            print(f'Reached iteration limit: {i}.')
            print(f'Time passed: {(time.time() - start_time) / 60:.2f} min.')
            break

        scheduler_it += 1

    optimizer.zero_grad()
    del optimizer
    return model, scheduler_it

def get_grad(model, train_loader,
             observation_variance=observation_variance,
             prior_variance=prior_variance, prior_weights=None):
    n_params = sum(p.numel() for p in model.parameters())
    grad_vector = torch.zeros(size=(n_params,), device=device)

    criterion = torch.nn.MSELoss(reduction='sum')

    n_batches = len(train_loader)

    for contexts, attention_mask, actions, rewards in train_loader:
        model.zero_grad()

        contexts = contexts.to(device)
        attention_mask = attention_mask.to(device)
        actions = actions.to(device)
        rewards = rewards.to(device=device, dtype=ptdtype)

        model_outputs = torch.gather(model(contexts, attention_mask=attention_mask), dim=1,
                                     index=actions.reshape((-1, 1)))

        loss_val = criterion(model_outputs, rewards.unsqueeze(dim=1).float()) / (
                2.0 * observation_variance)
        if isinstance(prior_variance, float):
            loss_val += torch.sum(
                torch.nn.utils.parameters_to_vector(model.parameters()) ** 2) / (
                                prior_variance * 2.0 * n_batches)
        else:
            loss_val += (0.5 / n_batches) * ((
                                                     torch.nn.utils.parameters_to_vector(
                                                         model.parameters()) - prior_weights) * prior_variance).dot(
                torch.nn.utils.parameters_to_vector(model.parameters()) - prior_weights)

        loss_val.backward()

        current_grad = torch.cat([p.grad.data.flatten()
                                  for p in model.parameters()])

        grad_vector += current_grad

    return grad_vector


def get_grad_last(model, train_loader,
             observation_variance=observation_variance,
             prior_variance=prior_variance, prior_weights=None):
    n_params = sum(p.numel() for p in model.parameters())
    grad_vector = torch.zeros(size=(n_params,), device=device)

    criterion = torch.nn.MSELoss(reduction='sum')

    n_batches = len(train_loader)

    for contexts, attention_mask, actions, rewards in train_loader:
        model.zero_grad()

        contexts = contexts.to(device)
        attention_mask = attention_mask.to(device)
        actions = actions.to(device)
        rewards = rewards.to(device=device, dtype=ptdtype)

        model_outputs = torch.gather(model(contexts, attention_mask=attention_mask), dim=1,
                                     index=actions.reshape((-1, 1)))

        loss_val = criterion(model_outputs, rewards.unsqueeze(dim=1).float()) / (
                2.0 * observation_variance)
        if isinstance(prior_variance, float):
            loss_val += torch.sum(
                torch.nn.utils.parameters_to_vector(model.parameters()) ** 2) / (
                                prior_variance * 2.0 * n_batches)
        else:
            raise ValueError

        loss_val.backward()

        current_grad = torch.cat([p.grad.data.flatten()
                                  for p in model.parameters()])

        grad_vector += current_grad

    return grad_vector





def training_loop_TS():
    global D
    D = []
    regret_trace = []

    model = bandit_model

    scheduler_it = 1

    print(f'N. of parameters: {sum(p.numel() for p in model.parameters())}')
    cumul = 0
    for i in tqdm(range(no_outer_loops)):
        gc.collect()
        torch.cuda.empty_cache()
        regret_cumsum = 0
        print('\n\ngathering new data...\n\n')
        for j in tqdm(range(no_inner_loops)):

            if i == 0:
                new_D, new_regret = generate_random_dataset(samples_per_inner_loop, get_regret=True, debug=DEBUG)
            else:
                new_D, new_regret = generate_dataset_sampling(samples_per_inner_loop, model,
                                                              updated_posterior_weights,
                                                              current_H, debug=DEBUG)
                gc.collect()
                torch.cuda.empty_cache()

            D += new_D
            regret_cumsum += new_regret

        regret_trace.append(regret_cumsum / (no_inner_loops * samples_per_inner_loop))
        cumul += regret_trace[-1]
        print('\n\n\n')
        print(f'round: {i}')
        print(f'avg: {cumul/(i+1)}')
        print(regret_trace)
        # model = train_model_from_dataset(model, new_D)
        new_train_loader = get_train_loader_from_dataset(new_D)

        if not incremental_training_enabled:
            print("Recreating model from scratch...")
            model = PretrainedBanditModelHF(init_from)
            model.to(device)

        if i == 0:
            updated_prior_precision = 1 / prior_variance
            updated_posterior_weights = copy.deepcopy(torch.nn.utils.parameters_to_vector(model.parameters()).detach())
            pointer = 0
            for name, param in model.named_parameters():
                curr_num_param = param.numel()
                if name == 'last_layer.weight' or name == 'last_layer.bias':
                    updated_posterior_weights[pointer:pointer + curr_num_param] = 0.0
                pointer += curr_num_param
            # print(updated_posterior_weights.dtype)
        else:
            updated_prior_precision = current_H

        model, scheduler_it = get_trained_model(model, new_train_loader, n_epochs=n_epochs, scheduler_it=scheduler_it,
                                                observation_variance=observation_variance,
                                                prior_precision=updated_prior_precision,
                                                prior_weights=updated_posterior_weights)

        gc.collect()
        torch.cuda.empty_cache()
        hessian_start_t = time.time()
        # if i == 0:
        #     prior_precision = 1 / prior_variance
        # else:
        #     prior_precision = current_H

        current_H = get_hessian(model, new_D,
                                observation_variance=observation_variance,
                                prior_precision=updated_prior_precision,
                                approx='ExpF',
                                subsample=None,
                                ).to(data_utils.hessian_ptdtype) # TODO WARNING!! USING BFLOAT16
        # current_H.to(ptdtype)
        # current_H.to(torch.bfloat16)
        print(f"Hessian data type: {current_H.dtype}")

        print(f'Time for H. computation: {(time.time() - hessian_start_t) / 60:.2f} min.')
        if updated_posterior_weights is not None:
            del updated_posterior_weights
            gc.collect()
            torch.cuda.empty_cache()

        updated_posterior_weights = copy.deepcopy(torch.nn.utils.parameters_to_vector(model.parameters()).detach())

        print(f'\nAvg precision: {torch.mean(current_H)}')
        print(f'Max precision: {torch.max(current_H)}')
        print(f'Min precision: {torch.min(current_H)}')
        # print(
        #    f'Only prior: {torch.count_nonzero(torch.isclose(current_H, torch.tensor([1 / prior_variance] * current_H.shape[0], device=device)))}')

    print(regret_trace)
    log_file.write(f'\ntotal regret: {sum(regret_trace)}\n')
    if print_action_selection:
        selected_actions = [d[2] for d in D]
        log_file.write('\nSelected actions: \n')
        log_file.write(str(selected_actions))
        print(f'selected actions: \n{selected_actions}')

    save_regret_trace(regret_trace, folder_name=generate_folder_name(config_dict))
    return regret_trace


def training_loop_epinet_TS():
    global D
    D = []
    regret_trace = []

    epinet_kwargs = {
        'z_dim': 32,
        'x_til_dim': 768,
        'hidden_epinet_dim': 256,
        'device': device,
    }

    model = NewPretrainedBanditModelEpiNetHF(init_from=init_from, epinet_class=Epinet5, epinet_kwargs=epinet_kwargs)
    model = model.to(device, dtype=ptdtype)

    scheduler_it = 1
    prior_precision = 1 / prior_variance
    prior_weights = copy.deepcopy(torch.nn.utils.parameters_to_vector(model.parameters()).detach())
    pointer = 0
    for name, param in model.named_parameters():
        curr_num_param = param.numel()
        if name == 'last_layer.weight' or name == 'last_layer.bias':
            prior_weights[pointer:pointer + curr_num_param] = torch.zeros(curr_num_param)
        pointer += curr_num_param

    print(f'N. of parameters: {sum(p.numel() for p in model.parameters())}')
    cumul = 0
    print("EPINETS TS")
    for i in tqdm(range(no_outer_loops)):
        regret_cumsum = 0
        print('\n\ngathering new data...\n\n')
        for j in tqdm(range(no_inner_loops)):

            if i == 0:
                new_D, new_regret = generate_random_dataset(samples_per_inner_loop, get_regret=True, debug=DEBUG)
            else:
                new_D, new_regret = generate_dataset_greedy(samples_per_inner_loop, model,
                                                            debug=DEBUG
                                                            )

            D += new_D
            regret_cumsum += new_regret

        regret_trace.append(regret_cumsum / (no_inner_loops * samples_per_inner_loop))
        cumul += regret_trace[-1]
        print('\n\n\n')
        print(f'round: {i}')
        print(f'avg: {cumul/(i+1)}')
        print(regret_trace)
        # model = train_model_from_dataset(model, new_D)
        new_train_loader = get_train_loader_from_dataset(new_D)

        if not incremental_training_enabled:
            print("Recreating model from scratch...")
            model = PretrainedBanditModelHF(init_from)
            model.to(device)


        if i == 0:
            updated_posterior_weights = prior_weights
        else:
            updated_posterior_weights = copy.deepcopy(torch.nn.utils.parameters_to_vector(model.parameters()).detach())



        model, scheduler_it = get_trained_model(model, new_train_loader, n_epochs=n_epochs, scheduler_it=scheduler_it,
                                                observation_variance=observation_variance,
                                                prior_precision=prior_precision,
                                                prior_weights=updated_posterior_weights,
                                                anneal_regul=False)



    print(regret_trace)

    save_regret_trace(regret_trace, folder_name=generate_folder_name(config_dict),  filename=f'{Epinet5.__name__}_regret_trace.pkl')
    log_file.write(f'\ntotal regret: {sum(regret_trace)}\n')

    return regret_trace

def training_loop_dropout_TS():
    global D
    D = []
    regret_trace = []

    model = bandit_model

    scheduler_it = 1
    prior_precision = 1 / prior_variance
    prior_weights = copy.deepcopy(torch.nn.utils.parameters_to_vector(model.parameters()).detach())

    pointer = 0
    for name, param in model.named_parameters():
        curr_num_param = param.numel()
        if name == 'last_layer.weight' or name == 'last_layer.bias':
            prior_weights[pointer:pointer + curr_num_param] = torch.zeros(curr_num_param)
        pointer += curr_num_param

    print(f'N. of parameters: {sum(p.numel() for p in model.parameters())}')
    cumul = 0
    print("DROPOUT TS")
    for i in tqdm(range(no_outer_loops)):
        regret_cumsum = 0
        print('\n\ngathering new data...\n\n')
        for j in tqdm(range(no_inner_loops)):

            if i == 0:
                new_D, new_regret = generate_random_dataset(samples_per_inner_loop, get_regret=True, debug=DEBUG)
            else:
                new_D, new_regret = generate_dataset_sampling_dropout(samples_per_inner_loop, model, debug=DEBUG,
                                                                      )

            D += new_D
            regret_cumsum += new_regret

        regret_trace.append(regret_cumsum / (no_inner_loops * samples_per_inner_loop))
        cumul += regret_trace[-1]
        print('\n\n\n')
        print(f'round: {i}')
        print(f'cumulative: {cumul}')
        print(regret_trace)
        # model = train_model_from_dataset(model, new_D)
        new_train_loader = get_train_loader_from_dataset(new_D)

        if not incremental_training_enabled:
            print("Recreating model from scratch...")
            model = PretrainedBanditModelHF(init_from)
            model.to(device)


        if i == 0:
            updated_posterior_weights = prior_weights
        else:
            updated_posterior_weights = copy.deepcopy(torch.nn.utils.parameters_to_vector(model.parameters()).detach())

        model, scheduler_it = get_trained_model(model, new_train_loader, n_epochs=n_epochs, scheduler_it=scheduler_it,
                                                observation_variance=observation_variance,
                                                prior_precision=prior_precision,
                                                prior_weights=updated_posterior_weights,
                                                anneal_regul=False)



    print(regret_trace)
    log_file.write(f'\ntotal regret: {sum(regret_trace)}\n')

    save_regret_trace(regret_trace, folder_name=generate_folder_name(config_dict),  filename='dropout_regret_trace.pkl')

    return regret_trace


def training_loop_last_layer():
    global D
    D = []
    regret_trace = []

    model = bandit_model
    n_last_layer_params = bandit_model.last_layer.weight.shape[0] * bandit_model.last_layer.weight.shape[1]
    scheduler_it = 1
    prior_weights = copy.deepcopy(torch.nn.utils.parameters_to_vector(model.parameters()).detach())
    pointer = 0
    for name, param in model.named_parameters():
        curr_num_param = param.numel()
        if name == 'last_layer.weight' or name == 'last_layer.bias':
            prior_weights[pointer:pointer + curr_num_param] = torch.zeros(curr_num_param)
        pointer += curr_num_param

    # prior_weights = None
    print(f'N. of parameters: {sum(p.numel() for p in model.parameters())}')
    cumul = 0
    for i in tqdm(range(no_outer_loops)):
        regret_cumsum = 0
        print('\n\ngathering new data...\n\n')
        for j in tqdm(range(no_inner_loops)):

            if i == 0:
                new_D, new_regret = generate_random_dataset(samples_per_inner_loop, get_regret=True, debug=DEBUG)
            else:
                new_D, new_regret = generate_dataset_sampling_last_layer(samples_per_inner_loop, model,
                                                                         current_H,
                                                                         debug=DEBUG)

            D += new_D
            regret_cumsum += new_regret

        regret_trace.append(regret_cumsum / (no_inner_loops * samples_per_inner_loop))
        cumul += regret_trace[-1]
        print('\n\n\n')
        print(f'round: {i}')
        print(f'avg: {cumul/(i+1)}')
        print(regret_trace)
        # model = train_model_from_dataset(model, new_D)
        new_train_loader = get_train_loader_from_dataset(new_D)

        if not incremental_training_enabled:
            print("Recreating model from scratch...")
            model = PretrainedBanditModelHF(init_from)
            model.to(device)

        if i == 0:
            updated_prior_precision = torch.diag(torch.tensor([1/prior_variance]*n_last_layer_params))
            updated_posterior_weights = prior_weights
        else:
            updated_prior_precision = current_H
            updated_posterior_weights = copy.deepcopy(torch.nn.utils.parameters_to_vector(model.parameters()).detach())

        model, scheduler_it = get_trained_model(model, new_train_loader, n_epochs=n_epochs, scheduler_it=scheduler_it,
                                                observation_variance=observation_variance,
                                                prior_precision=(1/prior_variance),
                                                # prior_weights=None)
                                                prior_weights=updated_posterior_weights,
                                                anneal_regul=False)

        hessian_start_t = time.time()
        # if i == 0:
        #     prior_precision = 1 / prior_variance
        # else:
        #     prior_precision = current_H

        current_H = get_hessian(model, new_D,
                                observation_variance=observation_variance,
                                prior_precision=updated_prior_precision,
                                approx='FullLast',
                                subsample=None,
                                )


        print(f'Time for H. computation: {(time.time() - hessian_start_t) / 60:.2f} min.')


        print(f'\nAvg precision: {torch.mean(current_H)}')
        print(f'Max precision: {torch.max(current_H)}')
        print(f'Min precision: {torch.min(current_H)}')
        # print(
        #    f'Only prior: {torch.count_nonzero(torch.isclose(current_H, torch.tensor([1 / prior_variance] * current_H.shape[0], device=device)))}')

    print(regret_trace)

    save_regret_trace(regret_trace, folder_name=generate_folder_name(config_dict), filename='regret_trace_last_layer.pkl')
    log_file.write(f'\ntotal regret: {sum(regret_trace)}\n')

    return regret_trace

def training_loop_greedy():
    global D
    D = []
    regret_trace = []

    model = bandit_model

    scheduler_it = 1
    prior_precision = (1/prior_variance)
    prior_weights = copy.deepcopy(torch.nn.utils.parameters_to_vector(model.parameters()).detach())
    pointer = 0
    for name, param in model.named_parameters():
        curr_num_param = param.numel()
        if name == 'last_layer.weight' or name == 'last_layer.bias':
            prior_weights[pointer:pointer + curr_num_param] = torch.zeros(curr_num_param)
        pointer += curr_num_param

    cumul = 0
    for i in tqdm(range(no_outer_loops)):
        regret_cumsum = 0
        print('gathering new data...')
        for j in tqdm(range(no_inner_loops)):
            # print(bootstrapped_model.linear3.weight)
            if i == 0:
                new_D, new_regret = generate_random_dataset(samples_per_inner_loop, get_regret=True, debug=DEBUG)
            else:
                new_D, new_regret = generate_dataset_greedy(samples_per_inner_loop, model, debug=DEBUG)
            D += new_D
            regret_cumsum += new_regret

        regret_trace.append(regret_cumsum / (no_inner_loops * samples_per_inner_loop))
        cumul += regret_trace[-1]
        print('\n\n\n')

        print(regret_trace[-300:])
        print(f'round: {i}')
        print(f'avg: {cumul/(i+1)}')
        print('\n')
        # model = train_model_from_dataset(model, new_D)
        # new_train_loader = get_train_loader_from_dataset(D)
        new_train_loader = get_train_loader_from_dataset(new_D)

        if not incremental_training_enabled:
            raise ValueError

        if i == 0:
            updated_posterior_weights = prior_weights
        else:
            updated_posterior_weights = copy.deepcopy(torch.nn.utils.parameters_to_vector(model.parameters()).detach())


        model, scheduler_it = get_trained_model(model, new_train_loader, n_epochs=n_epochs,
                                                scheduler_it=scheduler_it,
                                                observation_variance=observation_variance,
                                                prior_precision=prior_precision,
                                                prior_weights=updated_posterior_weights,
                                                anneal_regul=False)

    print(regret_trace)

    save_regret_trace(regret_trace, folder_name=generate_folder_name(config_dict), filename='regret_trace.pkl')
    log_file.write(f'\ntotal regret: {sum(regret_trace)}\n')

    return regret_trace


from tqdm import tqdm

regret_TS = []
regret_greedy = []

ts_map = {
    'last_la': training_loop_last_layer,
    'la': training_loop_TS,
    'epinet': training_loop_epinet_TS,
    'dropout': training_loop_dropout_TS,
}

if is_ts:
    training_loop = ts_map[args.ts]
    for _ in tqdm(range(repetitions)):
        regret_TS.append(training_loop())

    print("regret_TS:")
    print(regret_TS)
    print(np.array(regret_TS).mean(axis=0))

else:
    for _ in tqdm(range(repetitions)):
        regret_greedy.append(training_loop_greedy())

    print("regret_greedy:")
    print(regret_greedy)
    print(np.array(regret_greedy).mean(axis=0))

# print global properties
for name, value in config_dict.items():
    print(f'{name} = {value}')

log_file.close()
