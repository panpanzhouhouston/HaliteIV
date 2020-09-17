import numpy as np
import random
import copy
from collections import namedtuple, deque, defaultdict
import os

from model_v3 import Actor

import torch
import torch.nn.functional as F
from torch import nn as nn
from torch.autograd import Variable
import torch.optim as optim
#from torchsummary import summary


#BUFFER_SIZE = int(2e5)  # replay buffer size
BATCH_SIZE = 64        # minibatch size

LR_ACTOR = 5e-5        # learning rate of the actor
LR_DECAY = 0.9

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


"""
class WeightedCrossEntropyLoss(nn.Module):

    def __init__(self, ignore_index=-1):
        super(WeightedCrossEntropyLoss, self).__init__()
        self.ignore_index = ignore_index

    def forward(self, input, target):
        weight = self._class_weights(input)
        return F.cross_entropy(input, target, weight=weight, ignore_index=self.ignore_index)

    @staticmethod
    def _class_weights(input):
        # normalize the input first
        input = F.softmax(input, dim=1)
        flattened = flatten(input)
        nominator = (1. - flattened).sum(-1)
        denominator = flattened.sum(-1)
        class_weights = Variable(nominator / denominator, requires_grad=False)
        return class_weights
"""


def calc_loss(pred, target, metrics, weight):
    #pos_weight = torch.from_numpy(pos_weight).float().to(device)
    #criterion = torch.nn.BCEWithLogitsLoss(pos_weight=weight)
    criterion = nn.CrossEntropyLoss(weight)
    target = target.long()
    loss = criterion(pred, target)
    #print(f'loss: {loss.data.cpu()}')

    pred = torch.argmax(pred, dim=1)
    #print(f'target: {target.data.cpu().numpy()}')
    #print(f'pred: {pred.data.cpu().numpy()}')
    
    acc = np.sum(pred.data.cpu().numpy() == target.data.cpu().numpy()) / len(target.data.cpu().numpy())

    metrics['loss'] += loss.data.cpu().numpy()
    metrics['acc'] = acc

    return loss

class ActAgent():
    """Interacts with and learns from the environment."""
    
    def __init__(self, cin, vec_size, out_ship, out_shipyard, random_seed):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
        """
        self.cin = cin
        self.vec_size = vec_size
        self.out_ship = out_ship
        self.out_shipyard = out_shipyard
        self.seed = random.seed(random_seed)

        # Ship Actor Network
        self.actorship = Actor(cin, vec_size, out_ship, random_seed).to(device)
        self.actorship_optimizer = optim.Adam(self.actorship.parameters(), lr=LR_ACTOR)
        self.actorship_scheduler = optim.lr_scheduler.ExponentialLR(self.actorship_optimizer, gamma=LR_DECAY)
        
        # Shipyard Actor Network
        self.actorshipyard = Actor(cin, vec_size, out_shipyard, random_seed).to(device)
        self.actorshipyard_optimizer = optim.Adam(self.actorshipyard.parameters(), lr=LR_ACTOR)
        self.actorshipyard_scheduler = optim.lr_scheduler.ExponentialLR(self.actorshipyard_optimizer, gamma=LR_DECAY)
        # Actor Network summary
        # summary(self.actor_local, (self.state_size))

        # Replay memory
        self.shipmemory = ReplayBuffer(int(1e6), BATCH_SIZE, random_seed)
        self.shipyardmemory = ReplayBuffer(int(1e4), BATCH_SIZE, random_seed)
        
    def lr_step(self):
        """Schedule the learning rate, decay factor 0.99"""
        self.actorship_scheduler.step()
        self.actorshipyard_scheduler.step()
    
    def add(self, state1, state2, target, isShip=True):
         """Save experience in replay memory, and use random sample from buffer to learn."""
         if isShip:
             self.shipmemory.add(state1, state2, target)
         else:
             self.shipyardmemory.add(state1, state2, target)
         
    def step(self):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        # Save experience / reward
        #self.memory.add(state1, state2, action1_target, action2_target)
        # Learn, if enough samples are available in memory
        shipmetric_mean = defaultdict(float)
        shipyardmetric_mean = defaultdict(float)
        if len(self.shipmemory) > 1e5:
            for _ in range(100):
                experiences = self.shipmemory.sample()
                shipmetrics = self.learnship(experiences)
                for key, val in shipmetrics.items():
                    shipmetric_mean[key] += val
            for key, val in shipmetric_mean.items():
                shipmetric_mean[key] = val/100
        
        """
        for _ in range(100):
            experiences = self.shipyardmemory.sample()
            shipyardmetric = self.learnshipyard(experiences)
            for key, val in shipyardmetric.items():
                shipyardmetric_mean[key] += val
        for key, val in shipyardmetric_mean.items():
            shipyardmetric_mean[key] = val/100
        """
            
        return shipmetric_mean, shipyardmetric_mean

    def act(self, state1, state2, isShip=True):
        """Returns actions for given state as per current policy."""
        state1 = torch.from_numpy(state1).float().to(device)
        state2 = torch.from_numpy(state2).float().to(device)
        if isShip:
            self.actorship.eval()
            with torch.no_grad():
                out = self.actorship(state1, state2)
                out = out.cpu().data.numpy()
            self.actorship.train()
        else:
            self.actorshipyard.eval()
            with torch.no_grad():
                out = self.actorshipyard(state1, state2)
                out = out.cpu().data.numpy()
            self.actorshipyard.train()
        return out

    def learnship(self, experiences):
        """
        learn from replay experiencese
        """
        state1, state2, target = experiences
        state1 = state1.to(device)
        state2 = state2.to(device)
        target = target.to(device)

        # ---------------------------- update model ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        pred = self.actorship(state1, state2)
        # Minimize the loss
        metrics = defaultdict(float)
        #pos_weight = torch.tensor([1., 1., 1., 1., 0.5, 400., 2.5, 200.]).to(device)
        pos_weight = torch.tensor([1., 1., 1., 1., 0.75, 2000.]).to(device)
        #print(f'action1_pred dim:{action1_pred.shape}, action1_target dim:{action1_target.shape}')
        loss = calc_loss(pred, target, metrics, weight=pos_weight)
        
        # Minimize the loss
        self.actorship_optimizer.zero_grad()
        loss.backward()
        self.actorship_optimizer.step()
        
        return metrics
    
    def learnshipyard(self, experiences):
        """
        learn from replay experiencese
        """
        state1, state2, target = experiences
        state1 = state1.to(device)
        state2 = state2.to(device)
        target = target.to(device)

        # ---------------------------- update model ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        pred = self.actorshipyard(state1, state2)
        # Minimize the loss
        metrics = defaultdict(float)
        #pos_weight = torch.tensor([1., 1., 1., 1., 0.5, 400., 2.5, 200.]).to(device)
        pos_weight = torch.tensor([1., 80.]).to(device)
        #print(f'action1_pred dim:{action1_pred.shape}, action1_target dim:{action1_target.shape}')
        loss = calc_loss(pred, target, metrics, weight=pos_weight)
        
        # Minimize the loss
        self.actorshipyard_optimizer.zero_grad()
        loss.backward()
        self.actorshipyard_optimizer.step()
        
        return metrics
    
    def load_pretrained_weight(self, shipactor_file, shipyardactor_file):
        if os.path.isfile(shipactor_file):
            self.actorship.load_state_dict(torch.load(shipactor_file))
            
        if os.path.isfile(shipyardactor_file):
            self.actorshipyard.load_state_dict(torch.load(shipyardactor_file))


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state1", "state2", "target"])
        self.seed = random.seed(seed)
    
    def add(self, state1, state2, target):
        """Add a new experience to memory."""
        e = self.experience(state1, state2, target)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states1 = torch.from_numpy(np.stack([e.state1 for e in experiences if e is not None], axis=0)).float().to(device)
        states2 = torch.from_numpy(np.stack([e.state2 for e in experiences if e is not None], axis=0)).float().to(device)
        target = torch.from_numpy(np.stack([e.target for e in experiences if e is not None], axis=0)).float().to(device)

        return (states1, states2, target)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
