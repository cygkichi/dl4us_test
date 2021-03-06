import numpy as np
import networkx as nx
#import matplotlib.pyplot as plt
from graphviz import Digraph

def close_merge(A, edges):
    for i_from, i_to in edges:
        A[i_from, i_to] = 0
        A[i_from, i_from] = 1

def close_branch(A, edges):
    for i_from, i_to in edges:
        A[i_from, i_to] = 0

def entry_box(state):
    if np.random.rand()<0.3:
        state[:,0] = [0,1]
    if np.random.rand()<0.3:
        state[:,0] = [1,0]
    if np.random.rand()<0.3:
        state[:,16] = [0,1]
    if np.random.rand()<0.3:
        state[:,16] = [1,0]
    return state

def calc_reward(state):
    goalA = state[:,15]
    goalB = state[:,31]
    return goalA[0] + goalB[1] - 100*goalA[1] - 100*goalB[0]

class bucketbrigade(object):
    def __init__(self):
        self.n_box = 2
        self.setup_network()

    def setup_network(self):
        G = nx.DiGraph()
        G.add_path(range( 0,16))
        G.add_path(range(16,32))
        G.add_path([ 3,32,20]); G.add_path([23,33, 7]); G.add_path([12,34,28])
        self.G = G
        self.n_nodes = self.G.number_of_nodes()
        self.A = np.array(nx.adjacency_matrix(self.G).todense())
        branchs = [[[i,j] for j in list(v)] for i,v in self.G.adj._atlas.items() if len(v)>1]
        merges  = [[[j,i] for j in list(v)] for i,v in self.G.pred._atlas.items() if len(v)>1]
        self.branchs   = np.array(branchs)
        self.n_branchs = len(branchs)
        self.merges    = np.array(merges)
        self.n_merges  = len(merges)
        self.n_actions = np.array([len(x) for x in (branchs + merges)])

    def reset(self):
        init_state = np.zeros([self.n_box, self.n_nodes], dtype=np.int)
        init_state[0,0]  = 1
        init_state[0,10]  = 1
        init_state[1,7]  = 1
        init_state[1,16] = 1
        self.state = init_state
        return self.state.flatten()

    def step(self, action):
        a = '000000' + str(format(action, 'b'))
        a = a[-6:]
        action = [int(a[0]), int(a[1]), int(a[2]), int(a[3]), int(a[4]), int(a[5])]
        now_state    = self.state
        action_b = action[:self.n_branchs]
        action_m = action[self.n_branchs:]
        closedA  = np.copy(self.A)
        edges_b  = np.array([self.branchs[i][j] for i,j in enumerate(action_b)])
        close_branch(closedA, edges_b)
        edges_m  = np.array([self.merges[i][j] for i,j in enumerate(action_m)])
        close_merge( closedA, edges_m)
        is_stop   = np.sum(np.dot(now_state, closedA.T), axis=0) > 0
        stop_part = np.where(is_stop, now_state, 0)
        move_part = now_state - stop_part
        next_state = np.dot(move_part, closedA) + stop_part
        next_state = entry_box(next_state)
        reward   = calc_reward(next_state)
        if reward > -1:
            terminal = False
        else:
            terminal = True
        self.state = next_state
        return next_state.flatten(), reward, terminal, 1

    def make_image(self, output='graph'):
        G = Digraph(format='png')
        G.attr('node', shape='circle')
        edges = list(env.G.edges)
        for i in range(self.n_nodes):
            G.node(str(i), str(np.dot([1,2],self.state)[i]))
        for f,t in edges:
            G.edge(str(f), str(t))
        G.render(output)

if __name__ == '__main__':
    env = bucketbrigade()
    env.reset()
    action=[1,1,1,1,1,1]
    for i in range(50):
        no = '000' + str(i)
        no = no[-2:]
        env.make_image(output='graph_'+no)
        env.step(action)

