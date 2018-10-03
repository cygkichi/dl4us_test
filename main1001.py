#完成品　動作します
import numpy as np
import os
import numpy as np
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Input, Conv2D, Flatten
from keras.engine.topology import Layer
from keras.optimizers import RMSprop
from keras.utils import to_categorical
from agvsimulator_03 import *



def build_mlp():
    model = Sequential()
    model.add(Dense(100, activation='relu', input_shape=(405,)))
    model.add(Dense(2))
    model.compile(RMSprop(), 'mse')
    return model

class ReplayMemory:
    def __init__(self, memory_size):
        self.memory_size = memory_size
        self.memory = []

    def __len__(self):
        return len(self.memory)

    def append(self, transition):
        self.memory.append(transition)
        self.memory = self.memory[-self.memory_size:]

    def sample(self, batch_size):
        batch_indexes = np.random.randint(0, len(self.memory), size=batch_size).tolist()
        state      = np.array([self.memory[index]['state'] for index in batch_indexes])
        next_state = np.array([self.memory[index]['next_state'] for index in batch_indexes])
        reward     = np.array([self.memory[index]['reward'] for index in batch_indexes])
        action     = np.array([self.memory[index]['action'] for index in batch_indexes])
        terminal   = np.array([self.memory[index]['terminal'] for index in batch_indexes])
        return {'state': state, 'next_state': next_state, 'reward': reward, 'action': action, 'terminal': terminal}


def copy_weights(model_original, model_target):
    for i, layer in enumerate(model_original.layers):
        model_target.layers[i].set_weights(layer.get_weights())
    return model_target



#学習の初めはReplay Memoryが空で思い出すものが何もないため, 
#学習を始める前にランダムに行動した履歴をメモリに事前にためておきます.
memory_size = 10**6 # 10**6
initial_memory_size = 1000 # 50000
n_actions = 2
env = AGVSimulator()
replay_memory = ReplayMemory(memory_size)

step = 0    
n_steps = 1000
while True:
    state = env.reset().flatten()
    terminal = False

    while not terminal:
        action = np.random.randint(0, n_actions)
        next_state, reward, terminal, _ = env.step([int(i) for i in ('0000000'+format(action,'b'))[-1:]])
        next_state = np.clip(next_state.flatten(),0,1)
        #next_state = next_state.flatten()
        reward = np.sign(reward)
        transition = {
            'state': state,
            'next_state': next_state,
            'reward': reward,
            'action': action,
            'terminal': int(terminal)
        }
        replay_memory.append(transition)
        state = next_state
        step += 1
        if (step + 1) % 10000 == 0:
            print('Number of frames:', step + 1)
        if step > n_steps:
            terminal = 1
    if step >= initial_memory_size:
        break




#学習
#ネットワークの構築
model = build_mlp()
model_target = build_mlp()
eps_start = 1.0
eps_end = 0.2

gamma = 0.1
target_update_interval = 10
batch_size = 32
n_episodes = 1000

def get_eps(step):
    return max(eps_end, (eps_end - eps_start)/n_steps * step + eps_start)

def create_target(y, _t, action, n_actions):
    one_hot = to_categorical(action, n_actions)
    t = (1 - one_hot) * y + one_hot * _t[:, None]
    return t

def train(batch_size):
    batch = replay_memory.sample(batch_size)
    try:
        q = model.predict(batch['state'])
    except:
        from IPython.core.debugger import Pdb; Pdb().set_trace()
    q_target_next = model_target.predict(batch['next_state'])
    _t = batch['reward'] + (1 - batch['terminal']) * gamma * q_target_next.max(1)
    t = create_target(q, _t, batch['action'], n_actions)
    return model.fit(batch['state'], t, epochs=1, verbose=0)


for episode in range(n_episodes):
    state = env.reset().flatten()
    terminal = False
    total_reward = 0
    total_q_max = []
    step = 0
    while not terminal:
        q = model.predict(state.flatten()[None]).flatten()
        total_q_max.append(np.max(q))
        eps = get_eps(step)
        if np.random.random() < eps:
            action = np.random.randint(0, n_actions)
        else:
            action = np.argmax(q)
        next_state, reward, terminal, _ = env.step([int(i) for i in ('0000000'+format(action,'b'))[-1:]])
        next_state = np.clip(next_state.flatten(),0,1)
        #next_state = next_state.flatten()
        #reward = np.sign(reward)
        total_reward += reward
        #sprint(episode,step,total_reward,reward)
        transition = {
            'state': state,
            'next_state': next_state,
            'reward': reward,
            'action': action,
            'terminal': int(terminal)
        }
        replay_memory.append(transition)
        train(batch_size)
        state = next_state
        if (step + 1) % target_update_interval == 0:
            model_target = copy_weights(model, model_target)
        if step > n_steps:
            terminal = 1
        step += 1
#        print(step,'step')
    if (episode + 1) % 1 == 0:
        print('Episode: {}, Reward: {}, Q_max: {:.4f}, eps: {:.4f}, total_step: {}'.format(episode + 1, total_reward, np.mean(total_q_max), eps, step))

def test(N=100):
    state = env.reset()
    terminal = False
    total_reward = 0
    for i in range(N):
        #env.render()
        q = model.predict(state.flatten()[None]).flatten()
        action = np.argmax(q)
        print(env.state, action)
        next_state, reward, terminal, _ = env.step([int(i) for i in ('0000000'+format(action,'b'))[-1:]])
        next_state = np.clip(next_state.flatten(),0,1)
        #next_state = next_state.flatten()
        total_reward += reward
        state = next_state
    print(total_reward)

    
    
#test()
