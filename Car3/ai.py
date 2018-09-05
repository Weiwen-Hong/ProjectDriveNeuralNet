# AI for doom
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from kivy.clock import Clock
import time

import experience_replay

# Part 1 - Building the AI

# Making the brain
class CNN(nn.Module):
    
    def __init__(self, nb_action, nb_channels=1):
        super(CNN, self).__init__()
        # convolution connection
        self.cc1 = nn.Conv2d(in_channels = nb_channels, out_channels = 32, kernel_size = 5)
        # out_channels here means num of features to detect, kernel_size is dimension of feature detector
        self.cc2 = nn.Conv2d(in_channels = 32, out_channels = 32, kernel_size = 3)
        self.cc3 = nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 2)
        
        # full connection
        self.fc1 = nn.Linear(in_features = self.num_pixel((1,experience_replay.Box_size,experience_replay.Box_size)), out_features = 40)
        # infeature is num of pixel in picture
        self.fc2 = nn.Linear(in_features = 40, out_features = 40)
        self.fc3 = nn.Linear(in_features = 40, out_features = nb_action)
        self.optimizer = optim.Adam(self.parameters(), lr = 0.001)
        
    def num_pixel(self, img_dim):
        # prpagating fake image through the network to check how many pixel/ neuron
        x = Variable(torch.rand(1, *img_dim))
        
        # propagate through network
        x = F.relu(F.max_pool2d(self.cc1(x), 3, 2))
        x = F.relu(F.max_pool2d(self.cc2(x), 3, 2))
        x = F.relu(F.max_pool2d(self.cc3(x), 3, 2))
        
        return x.data.view(1, -1).size(1)
        
    def forward(self, x):
        # propagate through cc
        x = F.relu(F.max_pool2d(self.cc1(x), 3, 2))
        x = F.relu(F.max_pool2d(self.cc2(x), 3, 2))
        x = F.relu(F.max_pool2d(self.cc3(x), 3, 2))
        # flatten
        x = x. view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        
        return x
    
    def save(self):
        torch.save({'state_dic': self.state_dict(),
                    'optimizer': self.optimizer.state_dict()
                    }, 'last_brain.pth')

    def load(self):
        print("\n\n\n\n======================================================\n")
        if (os.path.isfile('last_brain.pth')):
            print('=> loading checkpoint...')
            checkpoint = torch.load('last_brain.pth')
            self.load_state_dict(checkpoint['state_dic'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            print('done!')
        else:
            print('No checkpoint found...')
        print("======================================================\n\n\n\n\n")
    
# Making the body
class SoftmaxBosy(nn.Module):
    
    def __init__(self, T):
        super(SoftmaxBosy, self).__init__()
        self.T = T
        
    def forward(self, outputs):
        probs = F.softmax(outputs * self.T)
        action = probs.multinomial()
        return action
    

# Making the AI
class AI:
    
    def __init__(self, brain, body):
        self.brain = brain
        self.body = body
        
    def __call__ (self, inputs):
        #start = time.time()
        
        x = Variable(torch.from_numpy(np.array(inputs, dtype = np.float32)))
        # conver to variable
        x = self.brain.forward(x)
        action = self.body.forward(x)
        
        #print("propagate: %f" %(time.time() - start))
        return action.data.numpy()
    

# Part 2 - Training AI with Deep Comvolutional Q-Learning



# Implementing Eligiblity trace algorithmn
def eligibility_trace(batch, cnn):
    gamma = 0.99
    inputs = []
    targets = []
    for series in batch:
        input = Variable(torch.from_numpy(np.array([series[0].state, series[-1].state], dtype = np.float32)))
        output = cnn(input)
        cumu_reward = 0.0 if series[-1].done else output[1].data.max()
        for step in reversed(series[:-1]):
            cumu_reward = step.reward + gamma * cumu_reward
        state = series[0].state
        target = output[0].data
        target[ series[0].action ] = cumu_reward
        
        inputs.append(state)
        targets.append(target)
    return torch.from_numpy(np.array(inputs, dtype = np.float32)), torch.stack(targets)
        
# Making the moving average on n steps
class MA:
    def __init__(self, size):
        self.list_of_reward = []
        self.size = size
        
    def add(self, reward):
        if isinstance(reward, list):
            self.list_of_reward += reward
        else:
            self.list_of_reward.append(reward)
        
        while len(self.list_of_reward) > self.size:
            del self.list_of_reward[0]
            
    def average(self):
        return np.mean(self.list_of_reward)
    
class CNN_AI:
    def __init__(self, map_env):
        self.num_action = 6   #  (left, right, stay_turn, speed up, speed down, stay_speed),  turn 10 degrees
        self.map_env = map_env
        self.game = self.map_env.parent
        
        # Building the AI
        self.cnn = CNN(self.num_action)
        self.softmaxBody = SoftmaxBosy(T = 1.0)
        self.ai = AI(brain = self.cnn, body = self.softmaxBody)
        
        # Setting up Experience Replay
        self.n_steps = experience_replay.NStepProgress(env = self.game, ai = self.ai, n_step = 10)
        self.mem = experience_replay.ReplayMemory(n_steps = self.n_steps, capacity = 10000)
        
        # movinfg average recorder of 100    
        self.ma = MA(100)
        
        # Training AI
        self.epoch = 1
        self.loss = nn.MSELoss()
        
        self.ai.brain.load()
        self.pause = True
        
    def learn(self, dt):
        if self.pause:
            return
        
        #start = time.time()
        self.mem.run_steps(20)
        #print("steps: %f" %(time.time() - start))
        
        #start = time.time()
        for batch in self.mem.sample_batch(10):
            inputs, targets = eligibility_trace(batch, self.cnn)
            inputs, targets = Variable(inputs), Variable(targets)
            predictions = self.cnn(inputs)
            loss_err = self.loss(predictions, targets)
            
            # optimize
            self.ai.brain.optimizer.zero_grad()
            loss_err.backward()
            self.ai.brain.optimizer.step()
            
        #print("learn: %f" %(time.time() - start))
            
        self.ai.brain.save()
        avg_steps = self.n_steps.rewards_steps()
        self.ma.add(avg_steps)
        avg_reward = self.ma.average()
        
        print("Epoch: %s, Average Reward: %s" %(str(self.epoch), str(avg_reward) ))
        self.epoch += 1
        
        if (avg_reward >= 10):
            print("Congratualations!! Your AI won.")
            
        self.ai.brain.save()