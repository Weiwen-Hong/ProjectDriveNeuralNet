# Experience Replay

# Importing the libraries
import numpy as np
from collections import namedtuple, deque
from kivy.clock import Clock
from torch.autograd import Variable
import time

# Defining one Step
Step = namedtuple('Step', ['state', 'action', 'reward', 'done'])
Box_size = 100
# Making the AI progress on several (n_step) steps

class NStepProgress:
    
    
    def __init__(self, env, ai, n_step):
        self.ai = ai
        self.rewards = []
        self.env = env
        self.n_step = n_step
        self.next_state = False
        self.r = False
        self.is_done = False
        self.action = False
        self.batches = []
                
    def step(self):
        
        history = deque()
        reward = 0.0
        state, _, _ = self.env.update(2, 360, Box_size)
        state = np.expand_dims(state, axis=0)
        
        #start = time.time()
        for i in range(self.n_step):
            #start = time.time()
            action = self.ai(np.array([state]))[0][0]
            #print("STEP 1: %f" %(time.time() - start))
            
            
            #start = time.time()
            next_state, r, is_done = self.env.update(action, 360, Box_size, True)
            #print("STEP 2 : %f\n\n\n\n\n\n\n\n\n" %(time.time() - start))
            
            
            reward += r
            history.append(Step(state = state, action = action, reward = r, done = is_done))
            state = next_state
            state = np.expand_dims(state, axis=0)
            
            self.rewards.append(reward)

        #print("STEP: %f" %(time.time() - start))
        return tuple(history)
        
    
    def rewards_steps(self):
        rewards_steps = self.rewards
        self.rewards = []
        return rewards_steps

# Implementing Experience Replay

class ReplayMemory:
    
    def __init__(self, n_steps, capacity = 10000):
        self.capacity = capacity
        self.n_steps = n_steps
        self.buffer = deque()

    def sample_batch(self, batch_size): # creates an iterator that returns random batches
        ofs = 0
        vals = list(self.buffer)
        np.random.shuffle(vals)
        while (ofs+1)*batch_size <= len(self.buffer):
            yield vals[ofs*batch_size:(ofs+1)*batch_size]
            ofs += 1

    def run_steps(self, samples):
        while samples > 0:
            #start = time.time()
            entry = self.n_steps.step() # 10 consecutive steps
            #print("STEP 1: %f" %(time.time() - start))
            
            #start = time.time()
            self.buffer.append(entry) # we put 200 for the current episode
            samples -= 1
            #print("STEP 2: %f\n\n\n\n\n\n" %(time.time() - start))
        while len(self.buffer) > self.capacity: # we accumulate no more than the capacity (10000)
            self.buffer.popleft()
