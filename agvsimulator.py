import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
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

def trade_cargo(state, cargo):
    pass

px = np.array([0,1,2,3,4,4,4,4,3,2,1,0,0,0,2,2])
py = np.array([0,0,0,0,0,1,2,3,3,3,3,3,2,1,1,2])

class AGVSimulator(object):
    def __init__(self):
        self.n_agvs  = 2
        self.t_cargo = 3
        self.setup_network()

    def setup_network(self):
        G = nx.DiGraph()
        G.add_path(range( 0,14))
        G.add_path([13, 0])
        G.add_path([ 2,14,15, 9])
        self.G = G
        N = G.number_of_nodes()
        self.n_nodes = N
        self.A = np.array(nx.adjacency_matrix(G).todense())
        branchs = [[[i,j] for j in list(v)] for i,v in G.adj._atlas.items()  if len(v)>1]
        merges  = [[[j,i] for j in list(v)] for i,v in G.pred._atlas.items() if len(v)>1]
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
        self.state = np.vstack([is_fill, cargo])
        return self.state

    def visualize(self, output='graph'):
        nodes = self.state[0] \
                + np.sum(self.state[1:], axis=0)
        cdic=np.array(['bx','ro','go'])
        for i in range(self.n_nodes):
            plt.plot(px[i], py[i], cdic[nodes[i]])
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
        is_stop = np.dot(now_state[0], closedA.T)
        stop_node = now_state * is_stop
        move_node = now_state - stop_node
        next_state = np.dot(move_node, closedA) + stop_node
        #next_state = trade_cargo(next_state,1)
        self.state = next_state
        reward = 1
        terminal = 1
        return next_state, reward, terminal, 1

if __name__ == '__main__':
    env = AGVSimulator()
    state = env.reset()
    env.visualize()


