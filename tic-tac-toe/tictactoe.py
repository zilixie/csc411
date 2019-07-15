from __future__ import print_function
from collections import defaultdict
from itertools import count
import numpy as np
import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributions
import matplotlib.pyplot as plt
from torch.autograd import Variable

class Environment(object):
    """
    The Tic-Tac-Toe Environment
    """
    # possible ways to win
    win_set = frozenset([(0,1,2), (3,4,5), (6,7,8), # horizontal
                         (0,3,6), (1,4,7), (2,5,8), # vertical
                         (0,4,8), (2,4,6)])         # diagonal
    # statuses
    STATUS_VALID_MOVE = 'valid'
    STATUS_INVALID_MOVE = 'inv'
    STATUS_WIN = 'win'
    STATUS_TIE = 'tie'
    STATUS_LOSE = 'lose'
    STATUS_DONE = 'done'

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset the game to an empty board."""
        self.grid = np.array([0] * 9) # grid
        self.turn = 1                 # whose turn it is
        self.done = False             # whether game is done
        return self.grid

    def render(self):
        """Print what is on the board."""
        map = {0:'.', 1:'x', 2:'o'} # grid label vs how to plot
        print(''.join(map[i] for i in self.grid[0:3]))
        print(''.join(map[i] for i in self.grid[3:6]))
        print(''.join(map[i] for i in self.grid[6:9]))
        print('====')

    def check_win(self):
        """Check if someone has won the game."""
        for pos in self.win_set:
            s = set([self.grid[p] for p in pos])
            if len(s) == 1 and (0 not in s):
                return True
        return False

    def step(self, action):
        """Mark a point on position action."""
        assert type(action) == int and action >= 0 and action < 9
        # done = already finished the game
        if self.done:
            return self.grid, self.STATUS_DONE, self.done
        # action already have something on it
        if self.grid[action] != 0:
            return self.grid, self.STATUS_INVALID_MOVE, self.done
        # play move
        self.grid[action] = self.turn
        if self.turn == 1:
            self.turn = 2
        else:
            self.turn = 1
        # check win
        if self.check_win():
            self.done = True
            return self.grid, self.STATUS_WIN, self.done
        # check tie
        if all([p != 0 for p in self.grid]):
            self.done = True
            return self.grid, self.STATUS_TIE, self.done
        return self.grid, self.STATUS_VALID_MOVE, self.done

    def random_step(self):
        """Choose a random, unoccupied move on the board to play."""
        pos = [i for i in range(9) if self.grid[i] == 0]
        move = random.choice(pos)
        return self.step(move)

    def play_against_random(self, action):
        """Play a move, and then have a random agent play the next move."""
        state, status, done = self.step(action)
        if not done and self.turn == 2:
            state, s2, done = self.random_step()
            if done:
                if s2 == self.STATUS_WIN:
                    status = self.STATUS_LOSE
                elif s2 == self.STATUS_TIE:
                    status = self.STATUS_TIE
                else:
                    raise ValueError("???")
        return state, status, done
        

class Policy(nn.Module):
    """
    The Tic-Tac-Toe Policy
    """
    def __init__(self, input_size=27, hidden_size=256, output_size=9):
        super(Policy, self).__init__()
        self.fullyConnected1 = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU()
        )
        self.fullyConnected2 = nn.Sequential(
            nn.Linear(hidden_size, output_size),
            nn.Softmax()
        )

    def forward(self, x):
        y = self.fullyConnected1(x)
        z = self.fullyConnected2(y)
        return z



def select_action(policy, state):
    """Samples an action from the policy at the state."""
    state = torch.from_numpy(state).long().unsqueeze(0)
    state = torch.zeros(3,9).scatter_(0,state,1).view(1,27)
    pr = policy(Variable(state))
    m = torch.distributions.Categorical(pr)
    action = m.sample()
    log_prob = torch.sum(m.log_prob(action))
    return action.data[0], log_prob
    
def compute_returns(rewards, gamma=1.0):
    """
    Compute returns for each time step, given the rewards
      @param rewards: list of floats, where rewards[t] is the reward
                      obtained at time step t
      @param gamma: the discount factor
      @returns list of floats representing the episode's returns
          G_t = r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + ... 

    >>> compute_returns([0,0,0,1], 1.0)
    [1.0, 1.0, 1.0, 1.0]
    >>> compute_returns([0,0,0,1], 0.9)
    [0.7290000000000001, 0.81, 0.9, 1.0]
    >>> compute_returns([0,-0.5,5,0.5,-10], 0.9)
    [-2.5965000000000003, -2.8850000000000002, -2.6500000000000004, -8.5, -10.0]
    """
    result = []
    G = []
    L = len(rewards)
    for j in range(L):
        G.append(gamma ** j)
    G = np.array(G)
    for i in range(L):
        R = np.array(rewards[i:])
        result.append(np.dot(R,G[0:len(R)]))
    return result
        


def finish_episode(saved_rewards, saved_logprobs, gamma=1.0):
    """Samples an action from the policy at the state."""
    policy_loss = []
    returns = compute_returns(saved_rewards, gamma)
    returns = torch.Tensor(returns)
    # subtract mean and std for faster training
    returns = (returns - returns.mean()) / (returns.std() +
                                            np.finfo(np.float32).eps)
    for log_prob, reward in zip(saved_logprobs, returns):
        policy_loss.append(-log_prob * reward)
    policy_loss = torch.cat(policy_loss).sum()
    policy_loss.backward(retain_graph=True)
    # note: retain_graph=True allows for multiple calls to .backward()
    # in a single step

def get_reward(status):
    """Returns a numeric given an environment status."""
    return {
            Environment.STATUS_VALID_MOVE: 10,
            Environment.STATUS_INVALID_MOVE: -60,
            Environment.STATUS_WIN: 80,
            Environment.STATUS_TIE: -20,
            Environment.STATUS_LOSE: -80
    }[status]
    

def game_simulation(env, policy, games=100, show=False):
    count_win,count_lose,count_tie = 0,0,0
    total,invalid = 0,0

    for i in range(games):
        state = env.reset()
        done = False
        while not done:
            action, logprob = select_action(policy, state)
            state, status, done = env.play_against_random(action)
            
            if show==True:
                env.render()
            total += 1
            if status == env.STATUS_INVALID_MOVE:
                invalid += 1
                

        if status == env.STATUS_WIN:
            count_win += 1
        if status == env.STATUS_LOSE:
            count_lose += 1
        if status == env.STATUS_TIE:
            count_tie += 1

    return count_win, count_lose, count_tie, total, invalid
    
    
def train(policy, env, gamma = 0.8, log_interval = 500, plot = False):
    """Train policy gradient."""
    optimizer = optim.Adam(policy.parameters(), lr = 0.0005)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size = 10000, gamma = 0.9)
    running_reward = 0
    episodes = []
    returns = []
    
    invalid_moves = []

    for i_episode in count(1):
        saved_rewards = []
        saved_logprobs = []
        state = env.reset()
        done = False
        while not done:
            action, logprob = select_action(policy, state)
            state, status, done = env.play_against_random(action)
            reward = get_reward(status)
            saved_logprobs.append(logprob)
            saved_rewards.append(reward)

        R = compute_returns(saved_rewards)[0]
        running_reward += R
        averge_return = running_reward/log_interval
        finish_episode(saved_rewards, saved_logprobs, gamma)

        if i_episode != 0 and i_episode % log_interval == 0:
            count_win, count_lose, count_tie, total, invalid = game_simulation(env, policy, 100)
            print('Episode: {}\tAverage return: {:.2f}\tInvalid: {}'.format(
                i_episode, averge_return, invalid))
            running_reward = 0
            episodes.append(i_episode)
            returns.append(averge_return)
            invalid_moves.append(invalid)

            torch.save(policy.state_dict(), "ttt/policy-%d.pkl" % i_episode)

        if i_episode % 1 == 0:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
        if i_episode == 50000:
            break

    if plot == True:
        # plot1, = plt.plot(episodes, returns,linestyle='--', color = "g")
        # plt.legend([plot1], ['average returns'])
        # plt.xlabel("episodes number")
        # plt.ylabel("average returns")
        # plt.title("training curve")
        # plt.savefig("./p5_training_curve.png")
        # # plt.show()
        
        #plt.clf()
        plot2, = plt.plot(episodes, invalid_moves,linestyle='--', color = "r")
        plt.legend([plot2], ['invalid moves'])
        plt.xlabel("episodes number")
        plt.ylabel("invalid moves")
        plt.title("training curve")
        plt.savefig("./p5c_training_curve.png")
        # plt.show()
    return

def p5b_diff_dim(policy, env, dim_list):
    #dim_list = [32, 64, 128, 256]
    table = list()
    for size in dim_list:
        env = Environment()
        policy = Policy(hidden_size = size)
        train(policy, env)
        counts = game_simulation(env, policy, games = 1000)
        table.append(counts)
    pair = (dim_list, table)
    for i in range(len(dim_list)):
        print("size= {}; win, lose, tie, total, invalid= {}".format(dim_list[i], table[i]))
    return pair

    

def p5d(policy, env):
    train(policy, env)
    count = game_simulation(env, policy, games = 100)
    print("win, lose, tie, total, invalid= {}".format(count))
    L = random.sample(range(1, 20), 5)
    for i in range(20):
        if i not in L:
            game_simulation(env, policy, games = 1)
        else:
            print("=======================\ngame: {}\n".format(i))
            game_simulation(env, policy, games = 1,show=True)
    return

    
def p6(env):
    episodes = []
    count_dict = {'win':[],'lose':[],'tie':[]}
    for i in range(1,51):
        episodes.append(1000*i)
        
    for episode in episodes:
        policy = Policy()
        load_weights(policy, episode)
        
        count_win,count_lose, count_tie,total,invalid = game_simulation(env, policy)
        ep_dict = {'win':count_win,'lose':count_lose,'tie':count_tie}
        for status in ['win','lose','tie']:
            count_dict[status].append(ep_dict[status])
        
    plot1, = plt.plot(episodes,count_dict['win'], linestyle='--',color = "c")
    plot2, = plt.plot(episodes,count_dict['lose'],linestyle=':', color = "m")
    plot3, = plt.plot(episodes,count_dict['tie'], linestyle='-.', color = "y")
    
    plt.legend([plot1,plot2,plot3], ["win", "lose", "tie"])
    plt.xlabel("episodes")
    plt.ylabel("games")
    plt.savefig("./part6.png")
    return
    
def add_moves(policy, env, M):
    first_distr = first_move_distr(policy, env)
    distr = np.array(first_distr)[0]
    for i in range(len(distr)):
        M[i].append(distr[i])
    
def part7(env):
    episodes = []
    for i in range(1,51):
        episodes.append(1000*i)
    moves = [[] for i in range(9)]
    for ep in episodes:
        policy = Policy()
        load_weights(policy, ep)
        add_moves(policy, env, moves)
        
    for i in range(9):
        plot, = plt.plot(episodes, moves[i])
        plt.ylabel("distribution")
        plt.xlabel("episodes")
        title = "move{}".format(i + 1)
        plt.legend([plot],[title])
        plt.title("P vs Episodes for move {}".format(i))
        plt.savefig("./figures/grid{}.png".format(i))
        plt.clf()
    return
    
    
def game_simulation_p8(env, policy, games=100):
    count_win,count_lose,count_tie = 0,0,0
    total,invalid = 0,0
    lose_games_list = []
    for i in range(games):
        state = env.reset()
        done = False
        print("Games: {}".format(i))
        while not done:
            action, logprob = select_action(policy, state)
            state, status, done = env.play_against_random(action)
            env.render()
            total += 1
            if status == env.STATUS_INVALID_MOVE:
                invalid += 1
                
        if status == env.STATUS_WIN:
            count_win += 1
        if status == env.STATUS_LOSE:
            count_lose += 1
            lose_games_list.append(i)
        if status == env.STATUS_TIE:
            count_tie += 1

    return count_win, count_lose, count_tie, total, invalid, lose_games_list 
    
    
def p8(policy, env):
    train(policy, env)
    count = game_simulation_p8(env, policy, games=100)
    print("win, lose, tie, total, invalid= {}".format(count))
    print(count[-1])
    

def first_move_distr(policy, env):
    """Display the distribution of first moves."""
    state = env.reset()
    state = torch.from_numpy(state).long().unsqueeze(0)
    state = torch.zeros(3,9).scatter_(0,state,1).view(1,27)
    pr = policy(Variable(state))
    return pr.data

def load_weights(policy, episode):
    """Load saved weights"""
    weights = torch.load("ttt/policy-%d.pkl" % episode)
    policy.load_state_dict(weights)

if __name__ == '__main__':
    import sys
    policy = Policy()
    env = Environment()
    

    # if len(sys.argv) == 1:
    #     # `python tictactoe.py` to train the agent
    #     train(policy, env)
    # else:
    #     # `python tictactoe.py <ep>` to print the first move distribution
    #     # using weightt checkpoint at episode int(<ep>)
    #     ep = int(sys.argv[1])
    #     load_weights(policy, ep)
    #     print(first_move_distr(policy, env))


