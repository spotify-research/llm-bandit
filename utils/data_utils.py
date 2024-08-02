import copy
import pickle
import random

import numpy as np
import torch
from torch.utils.data import RandomSampler, DataLoader, Dataset
from tqdm import tqdm
import os


'''
DATA HANDLING
'''

gradient_accumulation_steps = 1
# loader_batch_size: batch size of the data loader
loader_batch_size = 32



# data_dir = 'hf_data/imdb'
# data_dir = 'hf_data/tweet_eval/hate'
data_dir = 'hf_data/tweet_eval/offensive'
# data_dir = 'hf_data/toxic'

# block_size is the number of tokens for each data point
if 'imdb' in data_dir:
    block_size = 128
elif 'toxic' in data_dir:
    block_size = 256
elif 'tweet_eval' in data_dir:
    block_size = 64
else:
    raise ValueError(data_dir)

text_data = np.memmap(os.path.join(data_dir, 'data_text.bin'), dtype=np.uint16, mode='r').reshape(-1, block_size)
attention_data = np.memmap(os.path.join(data_dir, 'data_attention_mask.bin'), dtype=np.uint16, mode='r').reshape(-1, block_size)
label_data = np.memmap(os.path.join(data_dir, 'data_labels.bin'), dtype=np.uint16, mode='r')

n_rows = text_data.shape[0]

print(f'n rows: {n_rows}')

device = 'cuda'  # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.

dtype = 'float32'
# dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'  # 'float32' or 'bfloat16' or 'float16'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]

# hessian_dtype = 'bfloat16'
hessian_dtype = 'float32'
hessian_ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[hessian_dtype]

'''
We create a custom training dataset for our contextual bandit problem.
'''
class CustomTrainingDataset(Dataset):

    def __init__(self, D):
        self.D = copy.deepcopy(D)

    def __len__(self):
        return len(self.D)

    def __getitem__(self, idx):
        context, attention_mask, action, reward = self.D[idx]

        return torch.from_numpy(context.astype(np.int64)), \
               torch.from_numpy(attention_mask.astype(np.int64)), \
               action, \
               torch.tensor(reward, dtype=ptdtype)


# Reward function: asymmetric reward
def reward_fun(action, is_toxic):
    if is_toxic:
        # toxic
        best_outcome = 0.5
        if action == 0:
            return 0.5, best_outcome
        elif action == 1:
            return -0.5, best_outcome
    else:
        # non-toxic
        best_outcome = 1
        if action == 0:
            return 0.5, best_outcome
        elif action == 1:
            return 1, best_outcome


def sample_data_point():
    idx = random.randint(0, n_rows-1)
    return text_data[idx], label_data[idx], attention_data[idx]


def sample_data_batch(batch_size):
    idx = np.random.choice(n_rows, batch_size, replace=False)
    return text_data[idx], label_data[idx], attention_data[idx]


def generate_random_dataset(dataset_size, reward_fun=reward_fun, get_regret=False, debug=False):
    dataset = []
    regret = 0.0
    for _ in range(dataset_size):
        context_feature, is_toxic, attention_mask = sample_data_point()

        action = random.choice([0, 1])
        if debug:
            print(f'DEBUG: random action: {action}')

        reward, best_outcome = reward_fun(action, is_toxic)
        dataset.append((context_feature, attention_mask, action, reward))
        regret_increase = best_outcome - reward
        # print(f"Choosing action {action} in context {context}, momentary regret is {regret_increase}")
        regret += regret_increase

    if get_regret:
        return dataset, regret
    else:
        return dataset


def generate_dataset_sampling_linearized(dataset_size, original_model, prior_mean, H,
                                         reward_fun=reward_fun, ):
    dataset = []
    regret = 0.0
    selected_actions = []
    n_params = prior_mean.shape[0]
    new_J = torch.empty((2, n_params), device=device)

    stddev = (1 / H).sqrt()

    for i in tqdm(range(0, dataset_size, loader_batch_size)):

        if i + loader_batch_size <= dataset_size:
            current_batch_size = loader_batch_size
        else:
            current_batch_size = dataset_size - i

        context_batch, is_toxic_batch = sample_data_batch(current_batch_size)

        context_tensor_batch = torch.from_numpy(context_batch.astype(np.int64)).to(device) \
            .reshape(current_batch_size, -1)

        for j in range(current_batch_size):
            context_tensor, is_toxic = context_tensor_batch[j].reshape(1, -1), is_toxic_batch[j]

            # sample a model and take the best action
            samples = torch.randn(1, n_params, device=device)
            samples = samples * stddev
            predicted_weights = prior_mean.reshape(1, n_params) + samples

            def get_first(output):
                return output[0, 0]

            def get_second(output):
                return output[0, 1]

            # Jacobian computation: d model(x) / dw

            original_model.zero_grad()
            loss1 = get_first(original_model(context_tensor))
            loss1.backward()

            new_J[0] = torch.cat([p.grad.data.flatten()
                                  for p in original_model.parameters()])

            original_model.zero_grad()
            loss2 = get_second(original_model(context_tensor))
            loss2.backward()

            new_J[1] = torch.cat([p.grad.data.flatten()
                                  for p in original_model.parameters()])

            original_model.zero_grad()

            # predicted reward: J^T \dot (w_sampled - w_star) + model_{w_star} (x)
            first = torch.mm(new_J, (predicted_weights - prior_mean).T).T

            with torch.no_grad():
                second = original_model(context_tensor)

            predicted_reward = first + second
            # print(f'predicted reward: {predicted_reward.shape}')

            action = int(np.argmax(predicted_reward.detach().cpu().numpy()))

            reward, best_outcome = reward_fun(action, is_toxic)
            dataset.append((context_batch[j], action, reward))
            regret_increase = best_outcome - reward
            regret += regret_increase

    return dataset, regret

# The following function generates a dataset sampled from a model with Laplace Approximation
@torch.no_grad()
def generate_dataset_sampling(dataset_size, original_model, prior_mean, H,
                              reward_fun=reward_fun,
                              debug=False,
                              ):
    dataset = []
    regret = 0.0
    n_params = prior_mean.shape[0]
    original_model.eval()
    # stddev = (1 / H).sqrt() # TODO maybe waste of mem

    for i in tqdm(range(0, dataset_size, loader_batch_size)):

        if i + loader_batch_size <= dataset_size:
            current_batch_size = loader_batch_size
        else:
            current_batch_size = dataset_size - i

        context_batch, is_toxic_batch, attention_mask_batch = sample_data_batch(current_batch_size)

        context_tensor_batch = torch.from_numpy(context_batch.astype(np.int64)).to(device) \
            .reshape(current_batch_size, -1)

        attention_mask_tensor_batch = torch.from_numpy(attention_mask_batch.astype(np.int64)).to(device) \
            .reshape(current_batch_size, -1)

        for j in range(current_batch_size):
            context_tensor, is_toxic, attention_mask_tensor =\
                context_tensor_batch[j].reshape(1, -1), is_toxic_batch[j], attention_mask_tensor_batch[j]

            # # sample a model and take the best action
            # TODO WARNING!! USING BFLOAT16
            samples = torch.randn(1, n_params, device=device, dtype=ptdtype)
            # # samples = samples * stddev
            samples.mul_((1 / H).sqrt())
            # print(samples.dtype)
            # predicted_weights = prior_mean.reshape(1, n_params) + samples
            samples.add_(prior_mean.view(1, n_params))
            # print('DEBUG - Before:')
            # print(original_model.last_layer.weight)
            torch.nn.utils.vector_to_parameters(samples.view(-1), original_model.parameters())

            # TODO memory optimization
            # pointer = 0
            #
            # print('DEBUG - Before:')
            # print(original_model.last_layer.weight)
            # for params in original_model.parameters():
            #     curr_num_params = params.numel()
            #     samples = torch.randn(1, curr_num_params, device=device)
            #
            #     params.data = (prior_mean[pointer:pointer + curr_num_params] +
            #                       (samples * stddev[pointer: pointer + curr_num_params]))
            #     pointer += curr_num_params

            # print('DEBUG - After:')
            # print(original_model.last_layer.weight)

            # with torch.no_grad():
            predicted_reward = original_model(context_tensor, attention_mask=attention_mask_tensor)

            action = int(torch.argmax(predicted_reward).detach().cpu().numpy())
            if debug:
                print(f'DEBUG: pred. reward: {predicted_reward}')
                print(f'DEBUG: pred. action: {action}')
            reward, best_outcome = reward_fun(action, is_toxic)
            dataset.append((context_batch[j], attention_mask_batch[j], action, reward))
            regret_increase = best_outcome - reward
            regret += regret_increase



    # RESET ORIGINAL WEIGHTS
    torch.nn.utils.vector_to_parameters(prior_mean.reshape(-1), original_model.parameters())
    original_model.train()

    return dataset, regret


@torch.no_grad()
def generate_dataset_sampling_last_layer(dataset_size, original_model, H,
                              reward_fun=reward_fun,
                              debug=False,
                              ):
    dataset = []
    regret = 0.0

    last_layer_shape = original_model.last_layer.weight.shape
    n_params = last_layer_shape[0] * last_layer_shape[1]
    # covariance = torch.linalg.inv(H)

    for i in tqdm(range(0, dataset_size, loader_batch_size)):

        if i + loader_batch_size <= dataset_size:
            current_batch_size = loader_batch_size
        else:
            current_batch_size = dataset_size - i

        context_batch, is_toxic_batch, attention_mask_batch = sample_data_batch(current_batch_size)

        context_tensor_batch = torch.from_numpy(context_batch.astype(np.int64)).to(device) \
            .reshape(current_batch_size, -1)

        attention_mask_tensor_batch = torch.from_numpy(attention_mask_batch.astype(np.int64)).to(device) \
            .reshape(current_batch_size, -1)

        for j in range(current_batch_size):
            context_tensor, is_toxic, attention_mask_tensor =\
                context_tensor_batch[j].reshape(1, -1), is_toxic_batch[j], attention_mask_tensor_batch[j]

            # sample a model and take the best action
            # covariance = torch.linalg.inv(H)
            # samples = torch.from_numpy(
            #     np.random.multivariate_normal(mean=[0]*n_params, cov=covariance).reshape(last_layer_shape)
            #                            ).to(device)
            samples = torch.distributions.MultivariateNormal(torch.zeros(n_params, dtype=torch.float32),
                                                             precision_matrix=H).sample()\
                .reshape(last_layer_shape).to(device)


            original_model.last_layer.weight.add_(samples)
            original_model.eval()

            predicted_reward = original_model(context_tensor, attention_mask=attention_mask_tensor)

            original_model.last_layer.weight.sub_(samples)

            action = int(np.argmax(predicted_reward.detach().cpu().numpy()))

            if debug:
                print(f'DEBUG: pred. reward: {predicted_reward}')
                print(f'DEBUG: pred. action: {action}')

            reward, best_outcome = reward_fun(action, is_toxic)
            dataset.append((context_batch[j], attention_mask_batch[j], action, reward))
            regret_increase = best_outcome - reward
            regret += regret_increase

    original_model.train()
    return dataset, regret

@torch.no_grad()
def generate_dataset_sampling_last_layer_and_diag(dataset_size, original_model, prior_mean, diag_H, last_layer_H,
                                         reward_fun=reward_fun,
                                         ):
    dataset = []
    regret = 0.0
    n_params = prior_mean.shape[0]

    last_layer_shape = original_model.last_layer.weight.shape
    n_last_layer_params = last_layer_shape[0] * last_layer_shape[1]

    for i in tqdm(range(0, dataset_size, loader_batch_size)):

        if i + loader_batch_size <= dataset_size:
            current_batch_size = loader_batch_size
        else:
            current_batch_size = dataset_size - i

        context_batch, is_toxic_batch, attention_mask_batch = sample_data_batch(current_batch_size)

        context_tensor_batch = torch.from_numpy(context_batch.astype(np.int64)).to(device) \
            .reshape(current_batch_size, -1)

        attention_mask_tensor_batch = torch.from_numpy(attention_mask_batch.astype(np.int64)).to(device) \
            .reshape(current_batch_size, -1)

        for j in range(current_batch_size):
            context_tensor, is_toxic, attention_mask_tensor = \
                context_tensor_batch[j].reshape(1, -1), is_toxic_batch[j], attention_mask_tensor_batch[j]

            full_diag_samples = torch.randn(1, n_params, device=device, dtype=torch.bfloat16)
            full_diag_samples.mul_((1 / diag_H).sqrt())
            # print(samples.dtype)
            # predicted_weights = prior_mean.reshape(1, n_params) + samples
            full_diag_samples.add_(prior_mean.view(1, n_params))
            # print('DEBUG - Before:')
            # print(original_model.last_layer.weight)
            torch.nn.utils.vector_to_parameters(full_diag_samples.view(-1), original_model.parameters())

            last_layer_samples = torch.distributions.MultivariateNormal(prior_mean[-n_last_layer_params:],
                                                             precision_matrix=last_layer_H).sample() \
                .reshape(last_layer_shape).to(device)

            original_model.last_layer.weight = last_layer_samples


            original_model.eval()

            predicted_reward = original_model(context_tensor, attention_mask=attention_mask_tensor)


            action = int(np.argmax(predicted_reward.detach().cpu().numpy()))
            reward, best_outcome = reward_fun(action, is_toxic)
            dataset.append((context_batch[j], attention_mask_batch[j], action, reward))
            regret_increase = best_outcome - reward
            regret += regret_increase

    # RESET ORIGINAL WEIGHTS
    torch.nn.utils.vector_to_parameters(prior_mean.reshape(-1), original_model.parameters())
    original_model.train()
    return dataset, regret



def generate_dataset_sampling_dropout(dataset_size, original_model,
                              reward_fun=reward_fun, debug=False):

    def enable_dropout(model):
        """ Function to enable the dropout layers during test-time """
        for m in model.modules():
            if m.__class__.__name__.startswith('Dropout'):
                m.train()

    dataset = []
    regret = 0.0

    original_model.eval()
    enable_dropout(original_model)

    for i in tqdm(range(0, dataset_size, loader_batch_size)):

        if i + loader_batch_size <= dataset_size:
            current_batch_size = loader_batch_size
        else:
            current_batch_size = dataset_size - i

        context_batch, is_toxic_batch, attention_mask_batch = sample_data_batch(current_batch_size)

        context_tensor_batch = torch.from_numpy(context_batch.astype(np.int64)).to(device) \
            .reshape(current_batch_size, -1)

        attention_mask_tensor_batch = torch.from_numpy(attention_mask_batch.astype(np.int64)).to(device) \
            .reshape(current_batch_size, -1)

        for j in range(current_batch_size):
            context_tensor, is_toxic, attention_mask_tensor = \
                context_tensor_batch[j].reshape(1, -1), is_toxic_batch[j], attention_mask_tensor_batch[j]

            with torch.no_grad():
                predicted_reward = original_model(context_tensor, attention_mask=attention_mask_tensor)

            action = int(np.argmax(predicted_reward.detach().cpu().numpy()))

            if debug:
                print(f'DEBUG: pred. reward: {predicted_reward}')
                print(f'DEBUG: pred. action: {action}')

            reward, best_outcome = reward_fun(action, is_toxic)
            dataset.append((context_batch[j], attention_mask_batch[j], action, reward))
            regret_increase = best_outcome - reward
            regret += regret_increase

    original_model.train()

    return dataset, regret


def get_train_loader_from_dataset(D, replacement=False):
    if replacement:
        sampler = RandomSampler(D, replacement=True, num_samples=int(1e6))
        # n_epochs * loader_batch_size)

        train_dataloader = DataLoader(CustomTrainingDataset(D), batch_size=loader_batch_size,
                                      # shuffle=shuffle,
                                      sampler=sampler)
    else:
        train_dataloader = DataLoader(CustomTrainingDataset(D), batch_size=loader_batch_size,
                                      shuffle=True)

    return train_dataloader


def generate_dataset_greedy(dataset_size, original_model, reward_fun=reward_fun, return_selected_actions=False, debug=False):
    dataset = []
    regret = 0.0
    selected_actions = []
    original_model.eval()
    for _ in range(dataset_size):
        # random_word = sample_random_word()
        context_feature, is_toxic, attention_mask = sample_data_point()

        # take the best action in a greedy way

        predicted_reward = original_model(torch.from_numpy(context_feature.astype(np.int64)).to(device).reshape(1, -1),
                                          attention_mask=torch.from_numpy(attention_mask.astype(np.int64)).to(device).reshape(1, -1))

        action = int(torch.argmax(predicted_reward.detach().cpu()))

        if debug:
            print(f'DEBUG: pred. reward: {predicted_reward}')
            print(f'DEBUG: pred. action: {action}')

        selected_actions.append(action)

        reward, best_outcome = reward_fun(action, is_toxic)
        dataset.append((context_feature, attention_mask, action, reward))
        regret_increase = best_outcome - reward
        regret += regret_increase

    original_model.train()
    if return_selected_actions:
        return dataset, regret, selected_actions
    else:
        return dataset, regret


def save_regret_trace(regret_trace, folder_name='', filename='regret_trace.pkl'):
    """Save the regret trace list to a file."""

    # Ensure the folder exists
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    full_filename = os.path.join(folder_name, filename)

    # Check if the file already exists in the specified folder
    if os.path.exists(full_filename):
        # Load the existing regret trace
        with open(full_filename, 'rb') as file:
            existing_trace_list = pickle.load(file)

        # Append the new regret trace to the existing one
        existing_trace_list.append(regret_trace)
        regret_list_to_save = existing_trace_list
    else:
        regret_list_to_save = [regret_trace]

    # Save the combined regret trace
    with open(full_filename, 'wb') as file:
        pickle.dump(regret_list_to_save, file)

    print(f"Regret trace saved to {full_filename}")
