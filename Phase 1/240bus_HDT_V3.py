import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
import pickle
import random

from data_creater_240_ori import create_dataset
from mingpt_240_2.model_powersys_ori import HybridQuantumGPT, GPTQuantumConfig
from mingpt_240_2.train_powersys_ori import Trainer, TrainerConfig

seed = 63

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

data_dir_prefix = './240Bus_Graph_Data_V3_2/'#'./123Bus_Graph_Data_V2_1/' #'./123Bus_Graph_Data_V1_1/'
context_length = 1
model_type = 'reward_conditioned'
epochs = 2 # 1; 2
batch_size = 32
action_size = 23**5
model_eval = True

q_proj_qubits = 4
q_proj_layers = 1
q_sim_qubits = 3
q_sim_layers = 1
backend = "default.qubit"
use_quantum_similarity = True

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class StateActionReturnDataset(Dataset):

    def __init__(self, data, block_size, actions, done_idxs, rtgs, timesteps):
        self.block_size = block_size
        self.vocab_size = action_size
        self.data = data
        self.actions = actions
        self.done_idxs = done_idxs
        self.rtgs = rtgs
        self.timesteps = timesteps

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        block_size = self.block_size // 3
        done_idx = idx + block_size
        for i in self.done_idxs:
            if i > idx:  # first done_idx greater than idx
                done_idx = min(int(i), done_idx)
                break
        idx = done_idx - block_size
        states = torch.tensor(np.array(self.data[idx:done_idx]), dtype=torch.float32)
        actions = torch.tensor(self.actions[idx:done_idx], dtype=torch.long).unsqueeze(1)
        rtgs = torch.tensor(self.rtgs[idx:done_idx], dtype=torch.float32).unsqueeze(1)
        timesteps = torch.tensor(self.timesteps[idx:idx + 1], dtype=torch.int64).unsqueeze(1)

        return states, actions, rtgs, timesteps

def create_graph():
    edges = [[0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4,
              5, 5, 5, 6, 6, 6, 6, 6, 6, 7, 7, 7, 8, 8, 8, 8, 9, 9, 9, 9, 9,
              10, 10, 10, 11, 11, 11, 12, 12, 12, 12, 12, 13, 13, 13, 14, 14, 14,
              15, 15, 15, 15, 16, 16, 16, 16, 16, 17, 17, 18, 18, 18, 18,
              19, 19, 19, 20, 20, 21, 21, 21, 21, 22, 22],
             [0, 1, 2, 3, 0, 1, 4, 14, 0, 2, 6, 0, 3, 6, 11, 1, 4, 5, 10,
              4, 5, 9, 2, 3, 6, 7, 13, 16, 6, 7, 8, 7, 8, 9, 21, 5, 8, 9, 10, 15,
              4, 9, 10, 3, 11, 12, 11, 12, 13, 14, 15, 6, 12, 13, 1, 12, 14,
              9, 12, 15, 16, 6, 15, 16, 17, 18, 16, 17, 16, 18, 19, 21,
              18, 19, 20, 19, 20, 8, 18, 21, 22, 21, 22]]
    edge_index = torch.tensor(edges, dtype=torch.long)
    x = torch.eye(23)
    data = Data(x=x, edge_index=edge_index).to(device)
    return data

states, actions, rtg, traj_indx, timesteps = create_dataset(data_dir_prefix)

train_dataset = StateActionReturnDataset(states, context_length*3, actions, traj_indx, rtg, timesteps)

mconf = GPTQuantumConfig(train_dataset.vocab_size, train_dataset.block_size, state_size=len(states[0]) ,
                         n_layer=6, n_head=8, n_embd=128, model_type=model_type, max_timestep=max(timesteps),
                         q_proj_layers=q_proj_layers, q_proj_qubits=q_proj_qubits,
                         q_sim_layers=q_sim_layers, q_sim_qubits=q_sim_qubits,
                         backend=backend, use_quantum_similarity=use_quantum_similarity)
model = HybridQuantumGPT(mconf)

graph_data = create_graph()

tconf = TrainerConfig(max_epochs=epochs, batch_size=batch_size, learning_rate=6e-4,
                      lr_decay=True, warmup_tokens=512*20, final_tokens=2*len(train_dataset)*context_length*3,
                      num_workers=4, model_type=model_type, max_timestep=max(timesteps))
trainer = Trainer(model, train_dataset, None, tconf)

percnum = 20*1500
N = 50

if model_eval == False:
    trainer.train()
    torch.save(model.state_dict(), "model_HDT_240Bus_B_2.pt") # 1:_B_1; 2:_B_2
    print("Now evaluate model!")
    if model_type == 'naive':
        _ = trainer.evaluate(0)
    else:
        _ = trainer.evaluate(graph_data, percnum, N)
else:
    model.load_state_dict(torch.load("model_HDT_240Bus_B_2.pt")) # 1:_B_1; 2:_B_2
    if model_type == 'naive':
        trainer.evaluate(0)
    else:
        trainer.evaluate(graph_data, percnum, N)
