import gym
import numpy as np

class QLearningAgent(object):   
    def __init__(self):
        self.Q = []
        self.alpha = 1.0
        self.gamma = 0.8
        self.epsilon = 0.5
        self.env = gym.make("LunarLander-v2").env
        self.reward_list = []
        self.reflex = 1

    def DiscretizeState(self, state):
        param_list = []

        for i in range(len(state)):
            param_list.append(int(round(state[i] * 5)))

        return param_list

    def GetStateIndex(self, state):
        state_string = '['
        
        for param in state[:-1]:
            state_string += str(param) + ','

        state_string += str(state[-1]) + ']'

        return state_string
        
    def solve(self):
        ## Initialize Q(s, a) arbitrarily
        self.Q = {}

        ## Repeat (for each episode)
        episode = 1
        while(True):
            ## Initialize s
            s1 = self.env.reset()
            s1 = self.DiscretizeState(s1)
            s1_index = self.GetStateIndex(s1)
            a1 = np.random.randint(0, 4) if np.random.random(1) < self.epsilon else np.argmax(self.Q.get(s1_index, np.zeros(4)))
            
            episodic_reward = 0
            self.reflex = 1

            step_count = 0
            ## Loop for each step of episode:
            while(True):
                self.epsilon = 0.0
                
                ## Use reflex here
                if step_count % self.reflex == 0:
                    ## Choose A from S using policy derived from Q (e.g., "epsilon-greedy)
                    a1 = np.random.randint(0, 4) if np.random.random(1) < self.epsilon else np.argmax(self.Q.get(s1_index, np.zeros(4)))

                ## Take action A, observe R, S'
                s2, reward, done = self.env.step(a1)[:-1]
                step_count += 1
                s2 = self.DiscretizeState(s2)
                s2_index = self.GetStateIndex(s2)

                if step_count > 1000:
                    self.reflex = 3
                else:
                    self.reflex = 1

                if (reward > 50) and (done == True):
                    custom_reward = reward ** 7
                    print(custom_reward)
                    print(s1_index)
                    print(s2_index)
                else:
                    custom_reward = 0

                episodic_reward += reward

                temp_index_value = self.Q.get(s1_index, np.zeros(4))
                temp_index_value[a1] = temp_index_value[a1] + self.alpha * (reward + custom_reward + self.gamma * np.max(self.Q.get(s2_index, np.zeros(4))) - temp_index_value[a1])
                self.Q.update({s1_index:temp_index_value})

                if (reward > 50) and (done == True):
                    print(self.Q.get(s1_index, np.zeros(4)))

                s1 = s2

                if (done == True):
                    ## print("Next Episode")
                    self.reward_list.append(episodic_reward)
                    ## print(np.mean(np.array(self.reward_list)))
                    break
                
            if episode == 10000:
                print(self.Q)
                break

            episode += 1

agent = QLearningAgent()
agent.solve()
rewards = np.array(agent.reward_list)[len(agent.reward_list) - 200:]
print(np.mean(rewards))