import numpy as np
import networkx as nx
#import matplotlib.pyplot as plt
#import matplotlib.cm as cm
#eimport matplotlib.animation as animation
from graphviz import Digraph

def onehot(n, i):
    return np.eye(n,dtype=int)[i]

def close_merge(A, edges):
    for i, j in edges:
        A[i, j] = 0
        A[i, i] = 1

def close_branch(A, edges):
    for i,j in edges:
        A[i, j] = 0

def pick_cargo(state):
    new_state = []
    for i,s in enumerate(state.T):
        if (s[0]==1) and np.all(s[1:3]==0) and np.any(s[3:]==1):
            ns = [0] + list(s[3:]) + list(s[3:]*0)
        else:
            ns = list(s)
        new_state.append(ns)
    return np.array(new_state, dtype=int).T


px = np.array([ 0 , 1, 2, 3, 4, 5, 5, 5, 5, 5,\
                4 , 3, 2, 1, 0, 0, 0, 0, 2, 2,\
                2 ,12,13,14,15,16,17,17,17,17,\
                17,16,15,14,13,12,12,12,12,15,\
                15,15, 6, 7, 8, 9,10,10,10,10,\
                10,10,10,10,10,10,10,10,10,10,\
                10,11,11,10, 9, 8, 8, 8, 8, 8,\
                8 , 8, 8, 8, 8, 8, 8, 8, 8, 8,\
                7 , 6])
py = np.array([ 0 , 0, 0, 0, 0, 0, 1, 2, 3, 4,\
                4 , 4, 4, 4, 4, 3, 2, 1, 3, 2,\
                1 ,14,14,14,14,14,14,15,16,17,\
                18,18,18,18,18,18,17,16,15,17,\
                16,15, 1, 1, 1, 1, 1, 2, 3, 4,\
                5 , 6, 7, 8, 9,10,11,12,13,14,\
                15,15,17,17,17,17,16,15,14,13,\
                12,11,10, 9, 8, 7, 6, 5, 4, 3,\
                3 , 3])

class AGVSimulator(object):
    def __init__(self):
        self.n_agvs  = 5
        self.t_cargo = 2
        self.setup_network()

    def setup_network(self):
        G = nx.DiGraph()
        G.add_path(list(range( 0,18)) + [0] )
        G.add_path([12,18,19,20,2])
        G.add_path(list(range(21,39)) + [21])
        G.add_path([32,39,40,41,24])
        G.add_path([ 6]+list(range(42,62))+[38])
        G.add_path([36]+list(range(62,82))+[ 8])
        self.G = G
        N = G.number_of_nodes()
        self.n_nodes = N
        self.A = np.array(nx.adjacency_matrix(G).todense())
        branchs = [[[i,j] for j in list(v)] \
                   for i,v in G.adj._atlas.items()  if len(v)>1]
        merges  = [[[j,i] for j in list(v)] \
                   for i,v in G.pred._atlas.items() if len(v)>1]
        self.branchs   = np.array(branchs)
        self.n_branchs = len(branchs)
        self.merges    = np.array(merges)
        self.n_merges  = len(merges)
        self.n_actions = np.array([len(b) for b in (branchs + merges)])

    def reset(self, seed=None):
        na = self.n_agvs
        nn = self.n_nodes
        tc = self.t_cargo
        is_agv = np.zeros(nn ,dtype=int)
        is_agv[:na] = 1
        np.random.shuffle(is_agv)
        agv_state  = np.zeros([tc, nn],dtype=int)
        road_state = np.zeros([tc, nn],dtype=int)
        self.state = np.vstack([is_agv, agv_state, road_state])
        return self.state

    def animation(self,n_step=300):
        colors=np.array(['bx','ro','go'])
        fig = plt.figure()
        fig.patch.set_facecolor('black')
        ims = []
        for i in range(n_step):
            action = np.random.randint(2,size=len(self.n_actions))
            self.step(action)
            nodes = (np.argmax(self.state, axis=0)+1)*np.sum(self.state,axis=0)
            ax = plt.gca()
            ax.patch.set_facecolor('black')
            im = ax.scatter(px, py, c=nodes, marker='s',cmap=cm.hsv, vmax=6)
            ims.append([im])
        ani = animation.ArtistAnimation(fig, ims, interval=50)
        plt.show()


    def step(self, action):
        now_state = self.state
        action_b = action[:self.n_branchs]
        action_m = action[self.n_branchs:]
        closedA  = np.copy(self.A)
        edges_b  = np.array([self.branchs[i][j] for i,j in enumerate(action_b)])
        close_branch(closedA, edges_b)
        edges_m  = np.array([self.merges[i][j] for i,j in enumerate(action_m)])
        close_merge( closedA, edges_m)
        is_fill_agv = np.sum(self.state[:self.t_cargo+1], axis=0)>0
        is_stop = np.dot(is_fill_agv, closedA.T)
        is_stop = np.clip(1- now_state[:self.t_cargo+1] + is_stop,0,1)
        stop_agv = now_state[:self.t_cargo+1] * is_stop
        move_agv = now_state[:self.t_cargo+1] - stop_agv
        next_agv = np.dot(move_agv, closedA) + stop_agv
        next_state = np.vstack([next_agv, now_state[self.t_cargo+1:]])
        next_state  = pick_cargo(next_state)
        if np.all(next_state[:,16] == 0):
            if np.random.rand() < 0.1:
                next_state[3,16] = 1
        if np.all(next_state[:,19] == 0):
            if np.random.rand() < 0.1:
                next_state[4,19] = 1
        self.state = next_state
        reward = 0
        terminal = 0
        if self.state[1,28]==1:
            reward += 10
            self.state[0,28] = 1
            self.state[1,28] = 0
        if self.state[2,40]==1:
            reward += 10
            self.state[0,40] = 1
            self.state[2,40] = 0
        #if (self.state[2,28]==1) or (self.state[1,40]==1):
        #    terminal=1
        return next_state, reward, terminal, 0


if __name__ == '__main__':
    env = AGVSimulator()
    state = env.reset()


