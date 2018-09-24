import numpy as np
class mygame():
    def __init__(self):
        self.n_actions = 2
        self.n_nodes   = 12
        self.trans = np.zeros([self.n_actions ,self.n_nodes, self.n_nodes], dtype=np.int)
        self.trans[:,1,0]=1; self.trans[:,2,1]=1; self.trans[:, 4,3]=1; self.trans[:, 5, 4]=1
        self.trans[:,7,6]=1; self.trans[:,8,7]=1; self.trans[:,10,9]=1; self.trans[:,11,10]=1
        self.trans[0,3,2]=1; self.trans[0,9,8]=1
        self.trans[1,9,2]=1; self.trans[1,3,8]=1
        
    
    def reset(self):
        self.state = np.zeros(self.n_nodes, dtype = np.int)
        return self.state
    
    def step(self, action):
        now_state  = self.state
        next_state = np.dot( self.trans[action], now_state)
        next_state[0] = np.random.choice(11)
        next_state[6] = np.random.choice(11)
        reward = next_state[5] - next_state[11]
        if reward > -1:
            terminal = False
        else:
            terminal = True
        self.state = next_state
        return next_state, reward, terminal, 1
