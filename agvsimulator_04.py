import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.cm as cm
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

def pick_cargo(state):
    new_state = []
    for i,s in enumerate(state.T):
        if (s[0]>0) and np.all(s[1:3]==0) and np.any(s[3:]>0):
            ns = [0] + list(s[3:]) + list(s[3:]*0)
        else:
            ns = list(s)
        new_state.append(ns)
    return np.array(new_state, dtype=int).T


px = np.array([ 0 , 1, 2, 3, 4, 5, 6, 7, 8, 9,\
                10,11,12,13,14,14,14,14,14,14,\
                14,14,14,14,14,14,14,14,14,13,\
                12,11,10, 9, 8, 7, 6, 5, 4, 3,\
                2 , 1, 0, 0, 0, 0, 0, 0, 0, 0,\
                0 , 0, 0, 0, 0, 0, 1, 2, 3, 4,\
                4 ,13,12,11,10, 9, 8, 7, 6, 5,\
                4 , 3, 2, 1,13,12,11,11,11,11,\
                9 , 9, 9, 9, 9, 5, 5, 5, 5, 5,\
                11,11,11,11,11,11,11,12,13])
py = np.array([ 0 , 0, 0, 0, 0, 0, 0, 0, 0, 0,\
                0 , 0, 0, 0, 0, 1, 2, 3, 4, 5,\
                6 , 7, 8, 9,10,11,12,13,14,14,\
                14,14,14,14,14,14,14,14,14,14,\
                14,14,14,13,12,11,10, 9, 8, 7,\
                6 , 5, 4, 3, 2, 1, 2, 2, 2, 2,\
                1 , 8, 8, 8, 8, 8, 8, 8, 8, 8,\
                8 , 8, 8, 8,10,10,10,11,12,13,\
                13,12,11,10, 9, 9,10,11,12,13,\
                7 , 6, 5, 4, 3, 2, 1, 5, 5])

start_nodes=[ 0, 59]
goal_nodes =[76, 28]

class AGVSimulator(object):
    def __init__(self):
        self.n_agvs  = 10
        self.t_cargo = 2
        self.setup_network()

    def setup_network(self):
        G = nx.DiGraph()
        G.add_path(       list(range( 0,56)) + [ 0] )
        G.add_path([54] + list(range(56,61)) + [ 4])
        G.add_path([22] + list(range(61,74)) + [48])
        G.add_path([24] + list(range(74,80)) + [31])
        G.add_path([33] + list(range(80,85)) + [65])
        G.add_path([33] + list(range(80,85)) + [65])
        G.add_path([69] + list(range(85,90)) + [37])
        G.add_path([63] + list(range(90,97)) + [11])
        G.add_path([92] + list(range(97,99)) + [19])
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
        self.n_actions = np.array([len(b) for b in (branchs)])

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
            s = np.clip(self.state,0,1)
            nodes = (np.argmax(s, axis=0)+1)*np.sum(s,axis=0)
            ax = plt.gca()
            ax.patch.set_facecolor('black')
            im = ax.scatter(px, py, c=nodes, marker='s',cmap=cm.hsv, vmax=6)
            ims.append([im])
        ani = animation.ArtistAnimation(fig, ims, interval=50)
        plt.show()


    def step(self, action):
        now_state = self.state
        action_b = action
        action_m = np.random.randint(2, size=self.n_branchs)
        closedA  = np.copy(self.A)
        edges_b  = np.array([self.branchs[i][j] for i,j in enumerate(action_b)])
        close_branch(closedA, edges_b)
        edges_m  = np.array([self.merges[i][j] for i,j in enumerate(action_m)])
        close_merge( closedA, edges_m)
        is_fill_agv = np.max(self.state[:self.t_cargo+1], axis=0)>0
        is_stop = np.dot(is_fill_agv, closedA.T)
        is_stop = np.clip(1- np.clip(now_state[:self.t_cargo+1],0,1) + is_stop,0,1)
        stop_agv = now_state[:self.t_cargo+1] * is_stop
        move_agv = now_state[:self.t_cargo+1] * (1-is_stop)# stop_agv
        next_agv = np.dot(move_agv, closedA) + stop_agv
        next_state = np.vstack([next_agv, now_state[self.t_cargo+1:]])
        next_state  = pick_cargo(next_state)
        if np.all(next_state[:,start_nodes[0]] == 0):
            if np.random.rand() < 0.1:
                next_state[3,start_nodes[0]] = 1
        if np.all(next_state[:,start_nodes[1]] == 0):
            if np.random.rand() < 0.1:
                next_state[4,start_nodes[1]] = 1
        next_state = np.vstack([next_state[0],(next_state[1:]>0)*(next_state[1:]+1)])
        self.state = next_state
        reward = 0
        terminal = 0
        if self.state[1,goal_nodes[0]]>0:
            print(self.state[1,goal_nodes[0]],'goal1')
            reward += 1 #00 - self.state[1,30]
            self.state[0,goal_nodes[0]] = 1
            self.state[1,goal_nodes[0]] = 0
            #terminal = 1
        if self.state[2,goal_nodes[1]]>0:
            print(self.state[2,goal_nodes[1]],'goal2')
            reward += 1 #00 - self.state[2,75]
            self.state[0,goal_nodes[1]] = 1
            self.state[2,goal_nodes[1]] = 0


            #terminal = 1
#        if self.state[2,23]>0:
#            termianl = 1
#        if self.state[1,70]>0:
#            terminal = 1

#        print(self.state)]
#        print(np.max(self.state[1:]),'max')
#print(self.state)
        if np.max(self.state[1:]) > 100:
            terminal = 1
        return next_state, reward, terminal, 1


if __name__ == '__main__':
    env = AGVSimulator()
    state = env.reset()


