import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

def cut_branch(G, cut_edges):
    cutG = G.copy()
    for edge in cut_edges:
        cutG.remove_edge(edge[0], edge[1])
    return cutG

class bucketbrigade(object):
    def __init__(self):
        self.setup_network()

    def setup_network(self):
        G = nx.DiGraph()
        G.add_path(range( 0,16))
        G.add_path(range(16,32))
        G.add_path([ 3,32,20]); G.add_path([23,33, 7]); G.add_path([12,34,28])
        self.G = G
        N = self.G.number_of_nodes()
        self.n_nodes = N
        bs  = [[[i,v] for v in list(v) ] for i,v in self.G.adj._atlas.items() if len(v)>1]
        bs += [[[v,i] for v in list(v) ] for i,v in self.G.pred._atlas.items() if len(v)>1]
        self.branchs = np.array(bs)
        self.n_actions = np.array([len(b) for b in self.branchs])

    def reset(self):
        nn = self.n_nodes
        self.state = np.random.choice([0,1],nn)

    def step(self, action):
        now_state = self.state
        cut_edges = [self.branchs[i][j] for i,j in enumerate(action)]
        cutG = cut_branch(self.G, cut_edges)
        A = np.array(nx.adjacency_matrix(cutG).todense())
        is_stop = np.dot(now_state, A.T)
        stop_part = now_state * is_stop
        move_part = now_state - stop_part
        next_state = np.dot(move_part,A) + stop_part
        self.state = next_state
        print(now_state)
        print(move_part)
        print(stop_part)
        print(next_state)
        

if __name__ == '__main__':
    env = bucketbrigade()
    env.reset()
    
