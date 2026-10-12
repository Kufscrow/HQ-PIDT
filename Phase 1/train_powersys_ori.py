import math
import logging

from tqdm import tqdm
import numpy as np

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data.dataloader import DataLoader

logger = logging.getLogger(__name__)

from mingpt_240_2.utils_ori import sample
from F_240Bus_RLBase_Dyna import GraphEnv

#import atari_py
#from collections import deque
#import random
#import cv2
#import torch
#from PIL import Image


class TrainerConfig:
    # optimization parameters
    max_epochs = 10
    batch_size = 64
    learning_rate = 3e-4
    betas = (0.9, 0.95)
    grad_norm_clip = 1.0
    weight_decay = 0.1  # only applied on matmul weights
    # learning rate decay params: linear warmup followed by cosine decay to 10% of original
    lr_decay = False
    warmup_tokens = 375e6  # these two numbers come from the GPT-3 paper, but may not be good defaults elsewhere
    final_tokens = 260e9  # (at what point we reach 10% of original LR)
    # checkpoint settings
    ckpt_path = None
    num_workers = 1  # for DataLoader

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

class Trainer:
    def __init__(self, model, train_dataset, test_dataset, config):
        self.model = model
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.config = config

        # take over whatever gpus are on the system
        self.device = 'cpu'
        if torch.cuda.is_available():
            self.device = torch.cuda.current_device()
            self.model = torch.nn.DataParallel(self.model).to(self.device)

    def save_checkpoint(self):
        # DataParallel wrappers keep raw model object in .module attribute
        raw_model = self.model.module if hasattr(self.model, "module") else self.model
        logger.info("saving %s", self.config.ckpt_path)
        # torch.save(raw_model.state_dict(), self.config.ckpt_path)

    def train(self):
        model, config = self.model, self.config
        raw_model = model.module if hasattr(self.model, "module") else model
        optimizer = raw_model.configure_optimizers(config)

        self.tokens = 0
        model.train()

        #checkpoint_path = "./models/Inter_123_3/"

        for epoch in range(config.max_epochs):
            loader = DataLoader(self.train_dataset, shuffle=True,
                                batch_size=config.batch_size)

            losses = []
            pbar = tqdm(enumerate(loader), total=len(loader))
            for it, (x, y, r, t) in pbar:
            #for x, y, r, t in loader:
                # place data on the correct device
                #if it % 10 == 0 and epoch < 1 and it<=5060:
                #if it%100 == 0:
                #if it%200 == 0:
                #    checkpoint_file_path = checkpoint_path + 'model_' + str(epoch) + '_' + str(it) + '.pt'
                #    torch.save({
                #        'epoch': epoch,
                #        'update': it,
                #        'model_state_dict': model.state_dict()
                #    }, checkpoint_file_path)
                #x = x.to(self.device)
                #y = y.to(self.device)
                #r = r.to(self.device)
                #t = t.to(self.device)

                with torch.set_grad_enabled(True):
                    # logits, loss = model(x, y, r)
                    logits, loss = model(x, y, y, r, t)
                    loss = loss.mean()  # collapse all losses if they are scattered on multiple gpus
                    losses.append(loss.item())

                    # backprop and update the parameters
                    model.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)
                    optimizer.step()

                    if config.lr_decay:
                        self.tokens += (y >= 0).sum()  # number of tokens processed this step (i.e. label is not -100)
                        if self.tokens < config.warmup_tokens:
                            # linear warmup
                            lr_mult = float(self.tokens) / float(max(1, config.warmup_tokens))
                        else:
                            # cosine learning rate decay
                            progress = float(self.tokens - config.warmup_tokens) / float(
                                max(1, config.final_tokens - config.warmup_tokens))
                            lr_mult = max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))
                        lr = config.learning_rate * lr_mult
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr
                    else:
                        lr = config.learning_rate

                        # report progress
                    pbar.set_description(f"epoch {epoch+1} iter {it}: train loss {loss.item():.5f}. lr {lr:e}")

    def evaluate(self, graph_data, ret, N):
        model, config = self.model, self.config
        model.eval()

        env = GraphEnv(graph_data)

        T_rewards = []
        restored_power = []
        done = True
        error = False
        i = 0
        while i < N:
            print("Test start!")
            state = env.reset()
            binary_state = np.zeros(23)
            binary_state[state[:5]] = 1
            binary_state_tensor = torch.tensor(binary_state, dtype=torch.float32).to(self.device)
            binary_history_tensor = torch.tensor(env.binary_history, dtype=torch.float32).to(self.device)
            state_input = torch.cat((binary_state_tensor, binary_history_tensor)).reshape(1, 1, -1)

            rtgs = [ret]
            # first state is from env, first rtg is target return, and first timestep is 0
            sampled_action = sample(self.model.module, state_input, 1, temperature=1.0, sample=True, actions=None,
                                    rtgs=torch.tensor(rtgs, dtype=torch.float).to(self.device).unsqueeze(0).unsqueeze(
                                        -1),
                                    timesteps=torch.zeros((1, 1, 1), dtype=torch.int64).to(self.device))
            j = 0
            all_states = state_input
            actions = []
            reward_sum = 0
            while True:
                sampled_all_actions = []
                all_valid_actions = []
                action = num_to_twetri(sampled_action.cpu().numpy()[0, -1])
                for l in range(5):
                    valid_actions = env.get_valid_actions(l, state[l], sampled_all_actions)
                    all_valid_actions.append(valid_actions)
                    if action[l] in valid_actions:
                        sampled_all_actions.append(action[l])
                    else:
                        sampled_all_actions.append(state[l])
                actions += [torch.tensor(twetri_to_num(sampled_all_actions), dtype=torch.long).to(self.device).unsqueeze(
                                            0).unsqueeze(-1)]
                state, reward, done, errord, power = env.step(sampled_all_actions, all_valid_actions)
                reward_sum += reward
                j += 1

                if errord:
                    print("This is a failed attempt. System restarted")
                    break

                if done:
                    #if env.restoration_period == 10:
                    if env.restoration_period == 20:
                        T_rewards.append(reward_sum)
                        restored_power.append(power)
                        i += 1
                        print("This is the ", i, "th successful run.")
                    break

                binary_state = np.zeros(23)
                binary_state[state[:5]] = 1
                binary_state_tensor = torch.tensor(binary_state, dtype=torch.float32).to(self.device)
                binary_history_tensor = torch.tensor(env.binary_history, dtype=torch.float32).to(self.device)
                state_input = torch.cat((binary_state_tensor, binary_history_tensor)).reshape(1, 1, -1)

                all_states = torch.cat([all_states, state_input], dim=0)

                rtgs += [rtgs[-1] - reward]
                # all_states has all previous states and rtgs has all previous rtgs (will be cut to block_size in utils.sample)
                # timestep is just current timestep
                sampled_action = sample(self.model.module, all_states.unsqueeze(0), 1, temperature=1.0, sample=True,
                                        actions=torch.tensor(actions, dtype=torch.long).to(self.device).unsqueeze(
                                            1).unsqueeze(0),
                                        rtgs=torch.tensor(rtgs, dtype=torch.float).to(self.device).unsqueeze(
                                            0).unsqueeze(-1),
                                        timesteps=(min(j, self.config.max_timestep) * torch.ones((1, 1, 1),dtype=torch.int64).to(
                                            self.device)))
        eval_return = sum(T_rewards)/N
        print(T_rewards)
        print(restored_power)
        print("target return: %f, average eval return: %f with standard deviation %f" % (ret, eval_return, np.std(T_rewards)))
        print("Average eval restore power: %f with standard deviation %f" % (
        np.mean(restored_power), np.std(restored_power)))

def num_to_twetri(act):
    res = []
    ac = act
    while ac>0:
        r = ac%23
        res.append(r)
        ac //= 23
    if len(res)<5:
        l = len(res)
        for j in range(5-l):
            res.append(0)
    return res

def twetri_to_num(action):
    return int(sum([action[i] * (23 ** i) for i in range(len(action))]))
