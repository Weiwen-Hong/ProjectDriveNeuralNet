# Self Driving Car

# Importing the libraries
import numpy as np
from random import random, randint
import matplotlib.pyplot as plt
import time
import math

# Importing the Kivy packages
from kivy.app import App
from kivy.uix.widget import Widget
from kivy.uix.button import Button
from kivy.graphics import Color, Ellipse, Line
from kivy.config import Config
from kivy.properties import NumericProperty, ReferenceListProperty, ObjectProperty
from kivy.vector import Vector
from kivy.clock import Clock
from kivy.lang import Builder

import time
import thread
import ai


# Adding this line if we don't want the right click to put a red point
Config.set('input', 'mouse', 'mouse,multitouch_on_demand')

# Introducing last_x and last_y, used to keep the last point in memory when we draw the sand on the map
last_x = 0
last_y = 0
n_points = 0
length = 0 

# Getting our AI, which we call "brain", and that contains our neural network that represents our Q-function
action2rotation = [-10,-10, 0]
last_reward = 0
scores = []

# Initializing the map
first_update = True
def init():
    global sand
    global first_update
    sand = np.zeros((longueur,largeur))
    first_update = False

# Initializing the last distance
last_distance = 0

# Creating the car class

class Car(Widget):
    
    
    angle = NumericProperty(0)
    rotation = NumericProperty(0)
    velocity_x = NumericProperty(0)
    velocity_y = NumericProperty(0)
    velocity = ReferenceListProperty(velocity_x, velocity_y)
    sensor1_x = NumericProperty(0)
    sensor1_y = NumericProperty(0)
    sensor1 = ReferenceListProperty(sensor1_x, sensor1_y)
    sensor2_x = NumericProperty(0)
    sensor2_y = NumericProperty(0)
    sensor2 = ReferenceListProperty(sensor2_x, sensor2_y)
    sensor3_x = NumericProperty(0)
    sensor3_y = NumericProperty(0)
    sensor3 = ReferenceListProperty(sensor3_x, sensor3_y)
    signal1 = NumericProperty(0)
    signal2 = NumericProperty(0)
    signal3 = NumericProperty(0)



    def getPolarLidar(self, num, box_size):
        results = []
        pt = self.pos
        rotation = self.angle
       

        clamp = lambda n, minn, maxn: max(min(maxn, n), minn)
        half = int(math.floor(box_size / 2))

        for px in range(  clamp(int(pt[0] - half), 0, longueur)  , clamp(int(pt[0] + half), 0, longueur)  ):
            for py in range(   clamp(int(pt[1] - half), 0, largeur) , clamp(int(pt[1] + half), 0, largeur)):
                if sand[px][py] != 0:
                    angle = Vector(px,py).angle(rotation)
                    dist = Vector(px,py).distance(pt)

                    index = int(math.floor(angle/1))
                    
                    if results[index] > dist: 
                            results[index] = dist
        
        return results


    def getCoorLidar(self, box_size):
        result = np.zeros(shape=(box_size, box_size))
        half = int(math.floor(box_size / 2))
        
        n = range(box_size)
        for i in n:
            for j in n:
                x = int(self.pos[0] - half + i)
                y = int(self.pos[1] - half + j)
                
                # check if out range
                if (x < 0 or x >= longueur or y < 0 or y >= largeur):
                    result[i][j] = 1
                else:
                    result[i][j] = sand[x][y]
        return result
                


    def move(self, rotation, speed):
        self.rotation = rotation
        self.angle = self.angle + self.rotation
        
        if speed != 0:
            length = Vector(*self.velocity).length()
            scale = ( length + speed ) / length
            self.velocity[0]  = self.velocity[0] * scale
            self.velocity[1]  = self.velocity[1] * scale
        
        new_v = Vector(*self.velocity).rotate(self.rotation)
        self.velocity[0] = new_v.x
        self.velocity[1] = new_v.y
        
        self.pos = Vector(*self.velocity) + self.pos       
            

        self.sensor1 = Vector(30, 0).rotate(self.angle) + self.pos
        self.sensor2 = Vector(30, 0).rotate((self.angle+30)%360) + self.pos
        self.sensor3 = Vector(30, 0).rotate((self.angle-30)%360) + self.pos
        self.signal1 = int(np.sum(sand[int(self.sensor1_x)-10:int(self.sensor1_x)+10, int(self.sensor1_y)-10:int(self.sensor1_y)+10]))/400.
        self.signal2 = int(np.sum(sand[int(self.sensor2_x)-10:int(self.sensor2_x)+10, int(self.sensor2_y)-10:int(self.sensor2_y)+10]))/400.
        self.signal3 = int(np.sum(sand[int(self.sensor3_x)-10:int(self.sensor3_x)+10, int(self.sensor3_y)-10:int(self.sensor3_y)+10]))/400.
        if self.sensor1_x>longueur-10 or self.sensor1_x<10 or self.sensor1_y>largeur-10 or self.sensor1_y<10:
            self.signal1 = 1.
        if self.sensor2_x>longueur-10 or self.sensor2_x<10 or self.sensor2_y>largeur-10 or self.sensor2_y<10:
            self.signal2 = 1.
        if self.sensor3_x>longueur-10 or self.sensor3_x<10 or self.sensor3_y>largeur-10 or self.sensor3_y<10:
            self.signal3 = 1.

class Ball1(Widget):
    pass
class Ball2(Widget):
    pass
class Ball3(Widget):
    pass
class Dest(Widget):
    pass

# Creating the game class

class Game(Widget):

    car = ObjectProperty(None)
    ball1 = ObjectProperty(None)
    ball2 = ObjectProperty(None)
    ball3 = ObjectProperty(None)
    dest = ObjectProperty(None)
    

    def serve_car(self):
        self.car.center = self.center
        self.car.velocity = Vector(6, 0)

    def updateGoal(self, sec_num, box_size):
        # update goal
        polar = self.car.getPolarLidar(sec_num, box_size)
        index = np.argmax(polar)
        angle = np.pi * (index / sec_num)
        
        length = polar[index]
        if (length <= 0):
            length = 400
            
        clamp = lambda n, minn, maxn: max(min(maxn, n), minn)
        
        self.goal_x = np.cos(angle) * length + self.car.pos[0]
        self.goal_y = np.sin(angle) * length + self.car.pos[1]
        
        self.goal_x = clamp(self.goal_x, 0, longueur)
        self.goal_y = clamp(self.goal_y, 0, largeur)
        
        self.dest.pos = (int(self.goal_x), int(self.goal_y))
        print((int(self.goal_x), int(self.goal_y)))

    def update(self, action = 0, sec_num = 4, box_size = 10, enable = False):
        #print("inputs: A: %d, sec: %d, size: %d" %(action, sec_num, box_size))
        #print("updated, %d, %d\n\n\n\n" %(self.pos[0], self.pos[1]))
        
        #start = time.time()
        global brain
        global last_reward
        global scores
        global last_distance
        global longueur
        global largeur

        longueur = self.width
        largeur = self.height
        if first_update:
            init()
            self.goal_x = longueur / 2
            self.goal_y = largeur / 2
        
        value = action2rotation[action % 3]
        
        if True == True:
            if action < 3:
                self.car.move(value, 0)
            else:
                self.car.move(0, value)
        
        
        
        distance = np.sqrt((self.car.x - self.goal_x)**2 + (self.car.y - self.goal_y)**2)
        self.ball1.pos = self.car.sensor1
        self.ball2.pos = self.car.sensor2
        self.ball3.pos = self.car.sensor3

        if sand[int(self.car.x),int(self.car.y)] > 0:
            last_reward = -5
        else: # otherwise
            last_reward = -0.2
            if distance < last_distance:
                last_reward = 0.2
        

        if self.car.x < 10:
            self.car.x = 10
            last_reward = -1
        if self.car.x > self.width - 10:
            self.car.x = self.width - 10
            last_reward = -1
        if self.car.y < 10:
            self.car.y = 10
            last_reward = -1
        if self.car.y > self.height - 10:
            self.car.y = self.height - 10
            last_reward = -1
    
        if distance < 100:
            self.updateGoal(sec_num, box_size)
        last_distance = distance
        
        #print("ELSE: %f" %(time.time() - start))
        
        #start = time.time()
        #lidar = self.car.getCoorLidarParallel(box_size)
        #print("parallel: %f" %(time.time() - start))
        
        #start = time.time()
        lidar = self.car.getCoorLidar(box_size)
        #print("Serial:   %f\n\n\n\n\n\n\n" %(time.time() - start))
        
        return lidar, last_reward, False

# Adding the painting tools

class MyPaintWidget(Widget):

    def on_touch_down(self, touch):
        global length, n_points, last_x, last_y
        with self.canvas:
            Color(0.8,0.7,0)
            touch.ud['line'] = Line(points = (touch.x, touch.y), width = 10)
            last_x = int(touch.x)
            last_y = int(touch.y)
            n_points = 0
            length = 0
            sand[int(touch.x),int(touch.y)] = 1

    def on_touch_move(self, touch):
        global length, n_points, last_x, last_y
        if touch.button == 'left':
            touch.ud['line'].points += [touch.x, touch.y]
            x = int(touch.x)
            y = int(touch.y)
            length += np.sqrt(max((x - last_x)**2 + (y - last_y)**2, 2))
            n_points += 1.
            density = n_points/(length)
            touch.ud['line'].width = int(20 * density + 1)
            sand[int(touch.x) - 10 : int(touch.x) + 10, int(touch.y) - 10 : int(touch.y) + 10] = 1
            last_x = x
            last_y = y

# Adding the API Buttons (clear, save and load)

class CarApp(App):
    def __init__(self):
        super(CarApp, self).__init__()
        Builder.load_file('car.kv')

    def build(self):
        self.parent = Game()
        self.parent.serve_car()
        self.ai = ai.CNN_AI(self)
        self.painter = MyPaintWidget()
        
        clearbtn = Button(text = 'clear')
        savebtn = Button(text = 'save', pos = (self.parent.width, 0))
        loadbtn = Button(text = 'load', pos = (2 * self.parent.width, 0))
        pausebtn = Button(text = 'Pause', pos = (3 * self.parent.width, 0))
        
        clearbtn.bind(on_release = self.clear_canvas)
        savebtn.bind(on_release = self.save)
        loadbtn.bind(on_release = self.load)
        pausebtn.bind(on_release = self.pause)
        
        
        self.parent.add_widget(self.painter)
        self.parent.add_widget(clearbtn)
        self.parent.add_widget(savebtn)
        self.parent.add_widget(loadbtn)
        self.parent.add_widget(pausebtn)
    
        
        #Clock.schedule_interval(self.pauseCheck, 1/60)
        thread.start_new_thread(Clock.schedule_interval, (self.pauseCheck, 1/60))
        
        #thread.start_new_thread(Clock.schedule_interval, (self.loopAI, 1/60))
    
        return self.parent

    def pauseCheck(self, dt):
        if self.ai.pause == False:
            #self.ai.learn(dt)
            pass
        else:
            self.parent.update()
            
        
            
    def loopAI(self, dt):
        for i in range(0,100):
            self.ai.learn(0);
        
    def pause(self, obj):
        self.ai.pause = 1 - self.ai.pause
        
    def clear_canvas(self, obj):
        global sand
        self.painter.canvas.clear()
        sand = np.zeros((longueur,largeur))

    def save(self, obj):
        print("saving brain...")
        brain.save()
        plt.plot(scores)
        plt.show()

    def load(self, obj):
        print("loading last saved brain...")
        brain.load()
        
carApp = CarApp()
carApp.run()