import numpy as np
import datasets
import os
import tqdm
from transformers import GPT2Tokenizer
import argparse

# Set up argument parser
parser = argparse.ArgumentParser(description='Process dataset arguments.')
parser.add_argument('--data', choices=['offensive', 'hate', 'toxic', 'imdb'],
                    help='Type of dataset to use')
args = parser.parse_args()

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = 'left'

# Set dataset_args based on input argument
if args.data == 'offensive':
    dataset_args = ['cardiffnlp/tweet_eval', 'offensive']
elif args.data == 'hate':
    dataset_args = ['cardiffnlp/tweet_eval', 'hate']
elif args.data == 'toxic':
    dataset_args = ['ucberkeley-dlab/measuring-hate-speech', 'binary']
elif args.data == 'imdb':
    dataset_args = ['imdb']
else:
    raise ValueError(f'Invalid dataset type: {args.data}')


dataset_name = dataset_args[0]

dir_path = 'hf_data/'
if 'measuring-hate-speech' in dataset_name:
    block_size = 128
    dir_path += 'toxic'
elif 'imdb' in dataset_name:
    block_size = 256
    dir_path += 'imdb'
elif 'tweet_eval' in dataset_name:
    block_size = 64
    dir_path += 'tweet_eval/' + dataset_args[1]
else:
    raise ValueError(f'{dataset_name} not valid.')

def get_label(example, dataset_name):
    if 'measuring-hate-speech' in dataset_name:
        return example['hate_speech_score'] > 0.5
    elif 'imdb' in dataset_name:
        return 1 - example['label']
    elif 'tweet_eval' in dataset_name:
        return example['label']
    else:
        raise ValueError(f'{dataset_name} not valid.')



dataset = datasets.load_dataset(*dataset_args)


split_dataset = dataset['train']

tokenized = {'ids': [], 'is_toxic': [], 'attention_mask': []}

for i in tqdm.tqdm(range(len(split_dataset))):
    example = split_dataset[i]
    
    tokenized_example = tokenizer.encode_plus(
                        example['text'] + tokenizer.eos_token,                      # Sentence to encode.
                        max_length=block_size,           # Pad & truncate all sentences.
                        padding='max_length',
                        return_attention_mask = True,   # Construct attn. masks.
                        truncation = True,
                        # return_tensors = 'pt',     # Return pytorch tensors.
                   )

    tokenized['ids'].append(tokenized_example['input_ids'])
    tokenized['attention_mask'].append(tokenized_example['attention_mask'])
    tokenized['is_toxic'].append(np.uint16(get_label(example, dataset_name)))

X = np.array(tokenized['ids'], dtype=np.uint16)
attention_mask = np.array(tokenized['attention_mask'], dtype=np.uint16)
labels = np.array(tokenized['is_toxic'], dtype=np.uint16)

print(f'X shape: {X.shape}; \nAttention mask shape: {attention_mask.shape}; \n'
      f'labels shape: {labels.shape}.')

filename_X = 'data_text.bin'
filename_labels = 'data_labels.bin'
filename_attention_mask = 'data_attention_mask.bin'

os.makedirs(dir_path, exist_ok=True)

f_X = np.memmap(os.path.join(dir_path, filename_X), dtype=np.uint16, mode='w+', shape=X.shape)
f_mask = np.memmap(os.path.join(dir_path, filename_attention_mask), dtype=np.uint16, mode='w+', shape=attention_mask.shape)
f_labels = np.memmap(os.path.join(dir_path, filename_labels), dtype=np.uint16, mode='w+', shape=labels.shape)

f_X[:] = X[:]
f_mask[:] = attention_mask[:]
f_labels[:] = labels[:]

f_X.flush()
f_mask.flush()
f_labels.flush()
