import math
import numpy as np
import torch

# Learning rate scheduler
from torch.utils.data import DataLoader

from utils.data_utils import CustomTrainingDataset, loader_batch_size, ptdtype
from utils.vector_utils import vector_to_param_dict_last_layer
import random



device = 'cuda'

def get_lr(it, is_decayed_lr, warmup_iters, learning_rate, lr_decay_iters, min_lr):
    if is_decayed_lr:
        # 1) linear warmup for warmup_iters steps
        if it < warmup_iters:
            return learning_rate * it / warmup_iters
        # 2) if it > lr_decay_iters, return min learning rate
        if it > lr_decay_iters:
            return min_lr
        # 3) in between, use cosine decay down to min learning rate
        decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
        return min_lr + coeff * (learning_rate - min_lr)
    else:
        return learning_rate


def get_hessian(model,
                dataset,
                observation_variance,
                prior_precision,
                approx='ExpF',
                subsample=None):
    if approx == 'ExpF':
        return get_exp_f_hessian_3(model, dataset,
                                   observation_variance=observation_variance,
                                   prior_precision=prior_precision,
                                   subsample=subsample)
    elif approx == 'FullLast':
        return get_full_last_layer_hessian(model, dataset,
                                   observation_variance=observation_variance,
                                   prior_precision=prior_precision,
                                   subsample=subsample)
    elif approx == 'ExpF+FullLast':
        assert type(prior_precision) == tuple
        diag_prior_precision, last_layer_prior_precision = prior_precision
        return (get_exp_f_hessian_3(model, dataset,
                                   observation_variance=observation_variance,
                                   prior_precision=diag_prior_precision,
                                   subsample=subsample),
               get_full_last_layer_hessian(model, dataset,
                                    observation_variance=observation_variance,
                                    prior_precision=last_layer_prior_precision,
                                    subsample=subsample))
    else:
        raise NotImplementedError(f'Hessian approximation {approx} not implemented.')


def get_exp_f_hessian_3(model, dataset,
                        observation_variance,
                        prior_precision, subsample=None):
    H = get_exp_f_hessian_likelihood(model, dataset,
                                     observation_variance=observation_variance, subsample=subsample)
    H += prior_precision

    return H



def get_exp_f_hessian_likelihood(model, dataset,
                                 observation_variance, subsample=None):
    # n_params = len(torch.cat([p.grad.data.flatten() for p in model.parameters()]))
    n_params = sum(p.numel() for p in model.parameters())
    H = torch.zeros(size=(n_params,), device=device)

    dataset_len = len(dataset)

    if subsample is not None:
        subsampled_dataset = random.sample(dataset, min(subsample, dataset_len))
    else:
        subsampled_dataset = dataset

    hessian_loader = DataLoader(CustomTrainingDataset(subsampled_dataset),
                                batch_size=loader_batch_size,
                                shuffle=True)

    for contexts, attention_mask, actions, rewards in hessian_loader:

        contexts = contexts.to(device)
        attention_masks = attention_mask.to(device)
        actions = actions.to(device)

        def get_output(output, action):
            model_outputs = torch.gather(output, dim=1, index=action.reshape((-1, 1))).squeeze()
            return model_outputs

        for i in range(contexts.shape[0]):
            context, action = contexts[i].reshape(1, -1), actions[i].reshape(1, -1)
            attention_mask = attention_masks[i].reshape(1, -1)

            model.zero_grad()
            get_output(model(context, attention_mask=attention_mask), action).backward()
            H += torch.pow(torch.cat([p.grad.data.flatten()
                                      for p in model.parameters()]), 2)

    H /= observation_variance

    if subsample is not None:
        H *= (dataset_len / subsample)
    model.zero_grad()
    return H


def get_full_last_layer_hessian(model, dataset,
                        observation_variance,
                        prior_precision, subsample=None):
    H = get_full_last_layer_hessian_likelihood(model, dataset,
                                     observation_variance=observation_variance, subsample=subsample)
    H += prior_precision

    return H


def get_full_last_layer_hessian_likelihood(model, dataset,
                                 observation_variance, subsample=None):

    hessian_loader = DataLoader(CustomTrainingDataset(dataset),
                                batch_size=loader_batch_size,
                                shuffle=True)

    last_layer_shape = model.last_layer.weight.shape

    def h_loss(v_params):
        named_pd = vector_to_param_dict_last_layer(v_params, 'last_layer.weight', last_layer_shape)
        # print([p for p in model.named_parameters()][-1])
        # named_pd = vector_to_param_dict(v_params, [[p for p in model.named_parameters()][-1]])
        # print(named_pd)
        loss = 0.0
        for contexts, attention_mask, actions, rewards in hessian_loader:

            contexts = contexts.to(device)
            attention_masks = attention_mask.to(device)
            actions = actions.to(device)
            rewards = rewards.to(device=device, dtype=ptdtype)
            kwargs = {'attention_mask': attention_masks}

            out_fc = torch.func.functional_call(model, named_pd, contexts, kwargs=kwargs)
            model_outputs = torch.gather(out_fc, dim=1, index=actions.reshape((-1, 1))).squeeze()

            mseloss = torch.nn.MSELoss(reduction='sum')  # crucial to get right scale of the Hessian
            loss += mseloss(model_outputs, rewards.squeeze()) / (2.0 * observation_variance)

        return loss

    flattened_params_last_layer = torch.nn.utils.parameters_to_vector(model.last_layer.weight)

    H = torch.autograd.functional.hessian(h_loss, flattened_params_last_layer).detach()
    H = H.detach().cpu().numpy()
    H = 0.5 * (H.T + H)
    H_eval_min = np.linalg.eigvalsh(H).min()
    factor = 0.0 if H_eval_min >= 0.0 else abs(H_eval_min)

    Hpsd = H + factor * np.eye(H.shape[0])
    H = torch.tensor(Hpsd, dtype=ptdtype)

    H.to(device)
    return H
