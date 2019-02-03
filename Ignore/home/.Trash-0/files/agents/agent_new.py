import numpy as np
from task import Task

#BUILT ON CODE FROM https://github.com/keon/deep-q-learning/blob/master/dqn.py and policy_search.py
import random
import gym
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

class QLearning_Agent():
    def __init__(self, task):
        # Task (environment) information
        self.task = task
        self.state_size = task.state_size
        self.action_size = task.action_size
        self.action_low = task.action_low
        self.action_high = task.action_high
        self.action_range = self.action_high - self.action_low

        # Episode variables
        self.reset_episode()
        
        #QLearning attributes
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 0.1  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()
        self.batch_size = 32 #bstchsize for self.replay(batch_size)
    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate))
        return model
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    def act(self, state): #Picks rotor values
        print("in agent.act: " + str(state.shape))
        if np.random.rand() <= self.epsilon:
            print("less than epsilon")
            new_thrust = random.gauss(0., 1.)
            return [[new_thrust + random.gauss(0., 1.) for x in range(4)]]
        act_values = self.model.predict(state)
        return act_values 
    def replay(self): #Does the learning
        minibatch = random.sample(self.memory, self.batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * \
                    np.amax(self.model.predict(next_state)[0])
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay     
    def reset_episode(self):
        self.total_reward = 0.0
        self.count = 0
        state = self.task.reset()
        return state 
    def step(self, reward, done):
        # Save experience / reward
        self.total_reward += reward
        self.count += 1

        # Learn, if at end of episode
        if done and len(self.memory) > self.batch_size:
            print("DONE! RUNNING REPLAY ...")
            self.replay()
    def get_score(self):
        self.score = self.total_reward/self.count
        return self.score
            
            
            
            
            
            
            
            
           
        