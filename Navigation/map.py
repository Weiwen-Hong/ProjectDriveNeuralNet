# Self Driving Car

# Importing the libraries
import numpy as np
from random import random, randint
import matplotlib.pyplot as plt
import time


# Importing the Dqn object from our AI in ia.py
from ai import Dqn


# Introducing last_x and last_y, used to keep the last point in memory when we draw the sand on the map
last_x = 0
last_y = 0
n_points = 0 # the total number of points in the last drawing
length = 0 # the length of the last drawing

# Getting our AI, which we call "brain", and that contains our neural network that represents our Q-function
brain = Dqn(12,3,0.9) # 5 sensors, 3 actions, gama = 0.9
action2rotation = [0,10,-10] # action = 0 => no rotation, action = 1 => rotate 20 degres, action = 2 => rotate -20 degres
last_reward = 0 # initializing the last reward
scores = [] # initializing the mean score curve (sliding window of the rewards) with respect to time




# Creating the car class 

class Car:

    angle = 0 # initializing the angle of the car (angle between the x-axis of the map and the axis of the car)
    rotation = 0 # initializing the last rotation of the car (after playing the action, the car does a rotation of 0, 20 or -20 degrees)
    velocity = 0
    sensor1_x = 0 # initializing the x-coordinate of the first sensor (the one that looks forward)
    sensor1_y = 0 # initializing the y-coordinate of the first sensor (the one that looks forward)
    count1 = 0 # count of points within 2m
    count2 = 0 # count of points within 2m
    count3 = 0 # count of points within 2m
    count4 = 0 # count of points within 2m
    count5 = 0 # count of points within 2m
    count6 = 0 # count of points within 2m
    avg1 = 0 # average distance of points within a sector
    avg2 = 0 # average distance of points within a sector
    avg3 = 0 # average distance of points within a sector
    avg4 = 0 # average distance of points within a sector
    avg5 = 0 # average distance of points within a sector
    avg6 = 0 # average distance of points within a sector

    def rotate(self, rotation):
        # calls rotate method in cpp

    def changeSpeed(self, velocity):
        # calls set motor speed method in cpp
            

# Creating the game class
class Game:

    car = Car() # getting the car object from our kivy file
   

    def serve_car(self): # starting the car when we launch the application
        self.car.center = self.center # the car will start at the center of the map
        self.car.velocity = 6 # the car will start to go horizontally to the right with a speed of 6

    def update(self, close, count1, count2, count3, count4, count5, count6, avg1, avg2, avg3, avg4, avg5, avg6): # the big update function that updates everything that needs to be updated at each discrete time t when reaching a new state (getting new signals from the sensors)

        global brain # specifying the global variables (the brain of the car, that is our AI)
        global last_reward # specifying the global variables (the last reward)
        global scores # specifying the global variables (the means of the rewards)
        decision = [] # return the corresponding actions
    

        
        last_signal = [self.car.count1, self.car.count2, self.car.count3, self.car.count4, 
                       self.car.count5, self.car.count6, self.car.avg1, self.car.avg2, 
                       self.car.avg3, self.car.avg4, self.car.avg5, self.car.avg6] # our input state vector, composed of the three signals received by the three sensors, plus the orientation and -orientation
        action = brain.update(last_reward, last_signal) # playing the action from our ai (the object brain of the dqn class)
        scores.append(brain.score()) # appending the score (mean of the last 100 rewards to the reward window)
        rotation = action2rotation[action] # converting the action played (0, 1 or 2) into the rotation angle (0°, 20° or -20°)
        
        
        decision.append(rotation)
        

        if close > 5: # if the car is on the sand
            decision.append(1) # speed down to 1
            last_reward = -1 # and reward = -1
        else: # otherwise
            decision.append(6) # speed still 6
            last_reward = 1 # and it gets good reward 1

        return decision

 
            
class Navigation:

    parent = Game()
    parent.serve_car()

    def update(self, close, count1, count2, count3, count4, count5, count6, avg1, avg2, avg3, avg4, avg5, avg6):
        decision = self.parent.car.update(close, count1, count2, count3, count4, count5, count6, avg1, avg2, avg3, avg4, avg5, avg6)
        return decision
        
    def save(self, obj): # save button
        print("saving brain...")
        brain.save()
        plt.plot(scores)
        plt.show()

    def load(self, obj): # load button
        print("loading last saved brain...")
        brain.load()



