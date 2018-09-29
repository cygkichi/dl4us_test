import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.animation as animation
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

def pick_cargo(new_state, old_state):
    pass

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
        m  = self.t_cargo
        is_fill = np.zeros(nn ,dtype=int)
        is_fill[:na] = 1
        np.random.shuffle(is_fill)
        cargo = np.zeros([m,nn] ,dtype=int)
        cargo[1,1] = 1
        self.state = np.vstack([is_fill, cargo])
        return self.state

    def animation(self,n_step=100):
        colors=np.array(['bx','ro','go'])
        fig = plt.figure()
        fig.patch.set_facecolor('black')
        ims = []
        for i in range(n_step):
            action = np.random.randint(2,size=len(self.n_actions))
            self.step(action)
            nodes = self.state[0] + np.sum(self.state[1:],axis=0)
            ax = plt.gca()
            ax.patch.set_facecolor('black')
            im  = ax.scatter(px, py, c=nodes, marker='s')
            ims.append([im])
        ani = animation.ArtistAnimation(fig, ims, interval=200)
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
#        import pdb;pdb.set_trace()
        is_stop = np.dot(now_state[0], closedA.T)
        is_stop = np.clip(1- now_state[0] + is_stop,0,1)
        stop_node = now_state * is_stop
        move_node = now_state - stop_node
        next_state = np.dot(move_node, closedA) + stop_node
        #next_state = pick_cargo(next_state, old_state)
        self.state = next_state
        reward = 1
        terminal = 1
        return next_state, reward, terminal, 1

if __name__ == '__main__':
    env = AGVSimulator()
    state = env.reset()
    #env.visualize()


