from transformers import GPT2Model
import torch.nn as nn
import torch
import numpy as np
from transformers.activations import NewGELUActivation

from utils import data_utils


class PretrainedBanditModelHF(torch.nn.Module):
    def __init__(self, init_from):
        super().__init__()
        print(f"Initializing from OpenAI GPT-2 weights: {init_from}")
        print("Using HuggingFace implementation.")
        self.l1 = GPT2Model.from_pretrained(init_from)
        # self.l2 = nn.Dropout(0.3)

        last_layer_size_dict = {
            'gpt2': 768,
            'gpt2-medium': 1024,
            'gpt2-large': 1280,
            'gpt2-xl': 1600,
        }

        self.last_layer = nn.Linear(last_layer_size_dict[init_from], 2,
                            # bias=False
                            )

    def forward(self, input_ids, attention_mask, head_mask=None):
        out = self.l1(input_ids, attention_mask=attention_mask, head_mask=head_mask)
        # out = self.l2(out[0][:, -1, :])
        # print(out[0].shape)
        out = self.last_layer(out[0][:, -1, :])

        return out


class Epinet5(torch.nn.Module):
    def __init__(self, z_dim=32, x_til_dim=768, hidden_epinet_dim=256, device='cuda'):
        super().__init__()

        self.act = NewGELUActivation()
        self.device = device
        self.z_dim = z_dim
        self.hidden = nn.Linear(x_til_dim+z_dim, hidden_epinet_dim)
        self.last = nn.Linear(hidden_epinet_dim, 2*z_dim)

    def forward(self, x_til):
        batch_size = x_til.shape[0]
        z = torch.randn(size=(batch_size, self.z_dim), device=self.device, dtype=data_utils.ptdtype) / np.sqrt(self.z_dim)

        epinet_input = torch.cat([x_til, z], dim=1)
        out = epinet_input
        out = self.hidden(out)
        out = self.act(out)
        out = self.last(out)
        out = out.reshape((batch_size, 2, self.z_dim,))
        out = torch.matmul(out, z.reshape((batch_size, self.z_dim, 1)))
        return out.reshape((batch_size, 2))



class NewPretrainedBanditModelEpiNetHF(torch.nn.Module):
    def __init__(self, init_from, epinet_class, epinet_kwargs, epinet_prior=1.0):
        super().__init__()

        print(f"Initializing from OpenAI GPT-2 weights: {init_from}")
        print("Using HuggingFace implementation.")
        self.l1 = GPT2Model.from_pretrained(init_from)
        # self.l2 = nn.Dropout(0.3)

        self.epinet_prior = epinet_prior

        last_layer_size_dict = {
            'gpt2': 768,
            'gpt2-medium': 1024,
            'gpt2-large': 1280,
            'gpt2-xl': 1600,
        }

        self.last_layer = nn.Linear(last_layer_size_dict[init_from], 2,
                                    # bias=False
                                    )
        epinet_kwargs['x_til_dim'] = last_layer_size_dict[init_from]

        print(f"Initializing epinet: {str(epinet_class)}")
        self.prior_epinet = epinet_class(**epinet_kwargs).to(data_utils.ptdtype)
        self.learn_epinet = epinet_class(**epinet_kwargs).to(data_utils.ptdtype)


    def forward(self, input_ids, attention_mask):
        out = self.l1(input_ids, attention_mask=attention_mask)

        x_til = (out[0][:, -1, :]).detach()

        out_mean = self.last_layer(out[0][:, -1, :])

        out_learn_epinet = self.learn_epinet(x_til)

        with torch.no_grad():
            out_prior_epinet = self.prior_epinet(x_til)

        return out_mean + out_learn_epinet + self.epinet_prior * out_prior_epinet
