import gym
import numpy as np
import pickle
import math

class QLearningAgent(object):   
    def __init__(self, dumpPickle, pickle, alpha, gamma, epsilon_min, epsilon_max, training_episodes, BIN):
        self.Q = {}
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon_max
        self.epsilon_min = epsilon_min
        self.epsilon_max = epsilon_max
        self.reflex = 1
        self.episode = 0
        self.training_episodes = training_episodes
        self.BIN = BIN
        self.episodic_reward = []
        self.pickle = pickle
        self.dumpPickle = dumpPickle
        self.reward_list = []
        self.env = gym.make("LunarLander-v2").env

    def epsilon_decay(self):
        return max(
            self.epsilon_min,
            min(self.epsilon_max,
                1.0 - math.log10((self.episode + 1) / (self.training_episodes * 0.1))
            )
        )

    def CreateStateIndices(self, state):
        Q_indices = []
        state_value_meaning = ['x', 'y', 'dx', 'dy', 'theta', 'dtheta', 'legL', 'legR']
        combination_map = [
            [0, 1, 2, 3, 4, 5, 6, 7]
        ]
        for combination in combination_map:
            Q_index = ''
            for index in combination:
                Q_index += state_value_meaning[index] + ': ' + str(state[index])
            Q_indices.append(Q_index)
        return Q_indices
        
    def ChooseAction(self, state_1):
        mini_Q_matrix = []
        Q_indices = self.CreateStateIndices(state_1)

        for Q_index in Q_indices:
            mini_Q_matrix.append(self.Q.get(Q_index, np.zeros(4)))
        mini_Q_matrix = np.array(mini_Q_matrix)
        index = np.unravel_index(np.argmax(mini_Q_matrix), mini_Q_matrix.shape)

        self.epsilon = self.epsilon_decay()
        if np.random.random(1) < self.epsilon:
            action = np.random.randint(0, 4)
        else:
            action = index[1]
        return action

    def UpdateQTable(self, alpha, gamma, Qtable, state_1, state_2, reward, custom_reward, action):
        Q_indices_state_1 = self.CreateStateIndices(state_1)
        Q_indices_state_2 = self.CreateStateIndices(state_2)
        for s1_index, s2_index in zip(Q_indices_state_1, Q_indices_state_2):
            temp_index_value = Qtable.get(s1_index, np.zeros(4))
            temp_index_value[action] = temp_index_value[action] + alpha * (reward + custom_reward + gamma * np.max(Qtable.get(s2_index, np.zeros(4))) - temp_index_value[action])
            Qtable.update({s1_index:temp_index_value})

    ## 4, 4, 4, 4, 4, 4, 1, 1 works best so far
    def BucketizeState(self, state):
        param_list = []
        for sample, state_value in zip(self.BIN, state):
            param_list.append(int(round(state_value * sample)))
        return param_list
    
    def solve(self):
        try:
            if self.dumpPickle == True:
                with open(self.pickle, 'wb') as handle:
                    pickle.dump({}, handle, protocol=pickle.HIGHEST_PROTOCOL) 
            else:   
                with open(self.pickle, 'rb') as handle:
                    self.Q = pickle.load(handle)
        except:
            self.Q = {}

        self.episode = 1
        while(True):
            s1 = self.env.reset()
            s1 = self.BucketizeState(s1)
            a1 = self.ChooseAction(s1)
            
            episodic_reward = 0

            step_count = 0
            while(True):
                if self.episode % self.reflex == 0:
                    a1 = self.ChooseAction(s1)

                s2, reward, done = self.env.step(a1)[:-1]
                step_count += 1
                s2 = self.BucketizeState(s2)

                episodic_reward += reward

                self.reflex = 1
                if step_count > 1000:
                   self.reflex = 1000

                self.UpdateQTable(self.alpha, self.gamma, self.Q, s1, s2, reward, 0, a1)

                s1 = s2

                if (done == True):
                    self.reward_list.append(episodic_reward)
                    print('episode: ', self.episode, 'epsilon: ', self.epsilon, 'Reward of last 200 episodes: ', np.mean(self.reward_list[len(self.reward_list) - 100:]))
                    break
            
            if self.episode == self.training_episodes:
                print("Saving...")
                with open(self.pickle, 'wb') as handle:
                    pickle.dump(self.Q, handle, protocol=pickle.HIGHEST_PROTOCOL)
                    handle.close()
                break

            self.episode += 1
    
    ## reward_type = 'consecutive' or 'basic'
    def OutputTrainingData(self, reward_type):
        point_list = []
        episode_list = np.arange(1, len(self.reward_list) + 1)
        bin_list = ['BIN: [' + ''.join(str(bin_val) for bin_val in self.BIN) + ']'] * len(self.reward_list)

        if reward_type == 'consecutive':
            consecutive_value_list = []
            for i in range(0, len(self.reward_list)):
                if i < 100:
                    consecutive_value_list.append(np.mean(self.reward_list[ : i + 1]))
                else:
                    consecutive_value_list.append(np.mean(self.reward_list[i - 99 : i + 1]))
            point_dict = {'consecutive reward': consecutive_value_list, 'episode': episode_list, 'bin': bin_list}
            for value, episode, bin_value in zip(consecutive_value_list, episode_list, bin_list):
                point_list.append([value, episode, bin_value])
            specifier = ['consecutive reward', 'episode', 'bin']
            return point_dict, point_list, specifier

        elif reward_type == 'basic':
            point_dict = {'reward': self.reward_list, 'episode': episode_list, 'bin': bin_list}
            for value, episode, bin_value in zip(self.reward_list, episode_list, bin_list):
                point_list.append([value, episode, bin_value])
            specifier = ['reward', 'episode', 'bin']
            return point_dict, point_list, specifier

        else:
            return None, None