from model.actor import Actor
from model.critic import Critic
import torch
import numpy as np
from safetensors.torch import load_model, save_model
import os

actor = Actor()
critic = Critic()

print(os.path.join(os.getcwd(), 'standalone_examples/cartpole_train/ckpt/model.safetensors'))

save_model(actor, os.path.join(os.getcwd(), 'standalone_examples/cartpole_train/ckpt/model.safetensors'))
