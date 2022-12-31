from QLearningAgent import QLearningAgent
from Grapher import Grapher
import numpy as np
import pickle

def main():
    ## ------------------
    ## CONFIG SETTINGS
    ## ------------------
    SAVE_AGENT_ONE = False
    SAVE_AGENT_TWO = False
    SAVE_AGENT_THREE = False
    SAVE_AGENT_FOUR = False
    SAVE_AGENT_FIVE = False

    RUN_AGENT_ONE = False
    RUN_AGENT_TWO = False
    RUN_AGENT_THREE = False
    RUN_AGENT_FOUR = False
    RUN_AGENT_FIVE = False

    LOAD_AGENT_ONE = True
    LOAD_AGENT_TWO = True
    LOAD_AGENT_THREE = True
    LOAD_AGENT_FOUR = True
    LOAD_AGENT_FIVE = True
    ## ------------------

    if RUN_AGENT_ONE == True:
        agent = QLearningAgent(
            dumpPickle = True,
            pickle = 'QLearningTable.pickle',
            alpha = 1.0,
            gamma = 0.0,
            epsilon_min = 1.0,
            epsilon_max = 1.0,
            training_episodes = 5000,
            BIN = [4, 4, 4, 4, 4, 4, 1, 1]
        )

        no_of_runs = 0
        while(True):
            agent.solve()
            no_of_runs += 1

            if no_of_runs == 1:
                break

        point_dict_basic, point_list_basic, specifier_basic = agent.OutputTrainingData('basic')
        point_dict_consecutive, point_list_consecutive, specifier_consecutive = agent.OutputTrainingData('consecutive')

    if SAVE_AGENT_ONE == True:
        with open('PointDictBasic_Graph_13_14_15.pickle', 'wb') as handle:
            pickle.dump(point_dict_basic, handle, protocol=pickle.HIGHEST_PROTOCOL)
            handle.close()

        with open('PointListBasic_Graph_13_14_15.pickle', 'wb') as handle:
            pickle.dump(point_list_basic, handle, protocol=pickle.HIGHEST_PROTOCOL)
            handle.close()

        with open('SpecifierBasic_Graph_13_14_15.pickle', 'wb') as handle:
            pickle.dump(specifier_basic, handle, protocol=pickle.HIGHEST_PROTOCOL)
            handle.close()

        with open('PointDictConsecutive_Graph_13_14_15.pickle', 'wb') as handle:
            pickle.dump(point_dict_consecutive, handle, protocol=pickle.HIGHEST_PROTOCOL)
            handle.close()

        with open('PointListConsecutive_Graph_13_14_15.pickle', 'wb') as handle:
            pickle.dump(point_list_consecutive, handle, protocol=pickle.HIGHEST_PROTOCOL)
            handle.close()

        with open('SpecifierConsecutive_Graph_13_14_15.pickle', 'wb') as handle:
            pickle.dump(specifier_consecutive, handle, protocol=pickle.HIGHEST_PROTOCOL)
            handle.close()

    if LOAD_AGENT_ONE == True:
        try:
            with open('PointDictBasic_Graph_13_14_15.pickle', 'rb') as handle:
                point_dict_basic = pickle.load(handle)
                handle.close()

            with open('PointListBasic_Graph_13_14_15.pickle', 'rb') as handle:
                point_list_basic = pickle.load(handle)
                handle.close()

            with open('SpecifierBasic_Graph_13_14_15.pickle', 'rb') as handle:
                specifier_basic = pickle.load(handle)
                handle.close()

            with open('PointDictConsecutive_Graph_13_14_15.pickle', 'rb') as handle:
                point_dict_consecutive = pickle.load(handle)
                handle.close()

            with open('PointListConsecutive_Graph_13_14_15.pickle', 'rb') as handle:
                point_list_consecutive = pickle.load(handle)
                handle.close()

            with open('SpecifierConsecutive_Graph_13_14_15.pickle', 'rb') as handle:
                specifier_consecutive = pickle.load(handle)
                handle.close()
        except:
            print('No values loaded.')

    if (RUN_AGENT_ONE == True) or (LOAD_AGENT_ONE == True):
        ## Graph: Reward at each training episode while training your agent and discussion of results.
        grapher_1 = Grapher(point_dict_basic, point_list_basic, specifier_basic)
        grapher_1.Display_Graph(specifier_basic[0], specifier_basic[1], specifier_basic[2])

        ## Graph: Reward per episode for 100 consecutive episodes using you trained agent and discussion of
        ## the results.
        grapher_2 = Grapher(point_dict_consecutive, point_list_consecutive, specifier_consecutive)
        grapher_2.Display_Graph(specifier_consecutive[0], specifier_consecutive[1], specifier_consecutive[2])


    if RUN_AGENT_TWO == True:
        agent = QLearningAgent(
            dumpPickle = True,
            pickle = 'QLearningTable.pickle',
            alpha = 1.0,
            gamma = 0.0,
            epsilon_min = 1.0,
            epsilon_max = 1.0,
            training_episodes = 5000,
            BIN = [2, 2, 2, 2, 2, 2, 1, 1]
        )

        no_of_runs = 0
        while(True):
            agent.solve()
            no_of_runs += 1

            if no_of_runs == 1:
                break

        point_dict_basic_agent_two, point_list_basic_agent_two, specifier_basic_agent_two = agent.OutputTrainingData('basic')
        point_dict_consecutive_agent_two, point_list_consecutive_agent_two, specifier_consecutive_agent_two = agent.OutputTrainingData('consecutive')


    if SAVE_AGENT_TWO == True:
        with open('PointDictBasic_Graph_13_14_15_AGENT_TWO.pickle', 'wb') as handle:
            pickle.dump(point_dict_basic_agent_two, handle, protocol=pickle.HIGHEST_PROTOCOL)
            handle.close()

        with open('PointListBasic_Graph_13_14_15_AGENT_TWO.pickle', 'wb') as handle:
            pickle.dump(point_list_basic_agent_two, handle, protocol=pickle.HIGHEST_PROTOCOL)
            handle.close()

        with open('SpecifierBasic_Graph_13_14_15_AGENT_TWO.pickle', 'wb') as handle:
            pickle.dump(specifier_basic_agent_two, handle, protocol=pickle.HIGHEST_PROTOCOL)
            handle.close()

        with open('PointDictConsecutive_Graph_13_14_15_AGENT_TWO.pickle', 'wb') as handle:
            pickle.dump(point_dict_consecutive_agent_two, handle, protocol=pickle.HIGHEST_PROTOCOL)
            handle.close()

        with open('PointListConsecutive_Graph_13_14_15_AGENT_TWO.pickle', 'wb') as handle:
            pickle.dump(point_list_consecutive_agent_two, handle, protocol=pickle.HIGHEST_PROTOCOL)
            handle.close()

        with open('SpecifierConsecutive_Graph_13_14_15_AGENT_TWO.pickle', 'wb') as handle:
            pickle.dump(specifier_consecutive_agent_two, handle, protocol=pickle.HIGHEST_PROTOCOL)
            handle.close()

    if LOAD_AGENT_TWO == True:
        try:
            with open('PointDictBasic_Graph_13_14_15_AGENT_TWO.pickle', 'rb') as handle:
                point_dict_basic_agent_two = pickle.load(handle)
                handle.close()

            with open('PointListBasic_Graph_13_14_15_AGENT_TWO.pickle', 'rb') as handle:
                point_list_basic_agent_two = pickle.load(handle)
                handle.close()

            with open('SpecifierBasic_Graph_13_14_15_AGENT_TWO.pickle', 'rb') as handle:
                specifier_basic_agent_two = pickle.load(handle)
                handle.close()

            with open('PointDictConsecutive_Graph_13_14_15_AGENT_TWO.pickle', 'rb') as handle:
                point_dict_consecutive_agent_two = pickle.load(handle)
                handle.close()

            with open('PointListConsecutive_Graph_13_14_15_AGENT_TWO.pickle', 'rb') as handle:
                point_list_consecutive_agent_two = pickle.load(handle)
                handle.close()

            with open('SpecifierConsecutive_Graph_13_14_15_AGENT_TWO.pickle', 'rb') as handle:
                specifier_consecutive_agent_two = pickle.load(handle)
                handle.close()
        except:
            print('No values loaded.')

    if RUN_AGENT_THREE == True:
        agent = QLearningAgent(
            dumpPickle = True,
            pickle = 'QLearningTable.pickle',
            alpha = 1.0,
            gamma = 0.0,
            epsilon_min = 1.0,
            epsilon_max = 1.0,
            training_episodes = 5000,
            BIN = [10, 10, 10, 10, 10, 10, 1, 1]
        )

        no_of_runs = 0
        while(True):
            agent.solve()
            no_of_runs += 1

            if no_of_runs == 1:
                break

        point_dict_basic_agent_three, point_list_basic_agent_three, specifier_basic_agent_three = agent.OutputTrainingData('basic')
        point_dict_consecutive_agent_three, point_list_consecutive_agent_three, specifier_consecutive_agent_three = agent.OutputTrainingData('consecutive')


    if SAVE_AGENT_THREE == True:
        with open('PointDictBasic_Graph_13_14_15_AGENT_THREE.pickle', 'wb') as handle:
            pickle.dump(point_dict_basic_agent_three, handle, protocol=pickle.HIGHEST_PROTOCOL)
            handle.close()

        with open('PointListBasic_Graph_13_14_15_AGENT_THREE.pickle', 'wb') as handle:
            pickle.dump(point_list_basic_agent_three, handle, protocol=pickle.HIGHEST_PROTOCOL)
            handle.close()

        with open('SpecifierBasic_Graph_13_14_15_AGENT_THREE.pickle', 'wb') as handle:
            pickle.dump(specifier_basic_agent_three, handle, protocol=pickle.HIGHEST_PROTOCOL)
            handle.close()

        with open('PointDictConsecutive_Graph_13_14_15_AGENT_THREE.pickle', 'wb') as handle:
            pickle.dump(point_dict_consecutive_agent_three, handle, protocol=pickle.HIGHEST_PROTOCOL)
            handle.close()

        with open('PointListConsecutive_Graph_13_14_15_AGENT_THREE.pickle', 'wb') as handle:
            pickle.dump(point_list_consecutive_agent_three, handle, protocol=pickle.HIGHEST_PROTOCOL)
            handle.close()

        with open('SpecifierConsecutive_Graph_13_14_15_AGENT_THREE.pickle', 'wb') as handle:
            pickle.dump(specifier_consecutive_agent_three, handle, protocol=pickle.HIGHEST_PROTOCOL)
            handle.close()

    if LOAD_AGENT_THREE == True:
        try:
            with open('PointDictBasic_Graph_13_14_15_AGENT_THREE.pickle', 'rb') as handle:
                point_dict_basic_agent_three = pickle.load(handle)
                handle.close()

            with open('PointListBasic_Graph_13_14_15_AGENT_THREE.pickle', 'rb') as handle:
                point_list_basic_agent_three = pickle.load(handle)
                handle.close()

            with open('SpecifierBasic_Graph_13_14_15_AGENT_THREE.pickle', 'rb') as handle:
                specifier_basic_agent_three = pickle.load(handle)
                handle.close()

            with open('PointDictConsecutive_Graph_13_14_15_AGENT_THREE.pickle', 'rb') as handle:
                point_dict_consecutive_agent_three = pickle.load(handle)
                handle.close()

            with open('PointListConsecutive_Graph_13_14_15_AGENT_THREE.pickle', 'rb') as handle:
                point_list_consecutive_agent_three = pickle.load(handle)
                handle.close()

            with open('SpecifierConsecutive_Graph_13_14_15_AGENT_THREE.pickle', 'rb') as handle:
                specifier_consecutive_agent_three = pickle.load(handle)
                handle.close()
        except:
            print('No values loaded.')


    if RUN_AGENT_FOUR == True:
        agent = QLearningAgent(
            dumpPickle = True,
            pickle = 'QLearningTable.pickle',
            alpha = 1.0,
            gamma = 0.0,
            epsilon_min = 1.0,
            epsilon_max = 1.0,
            training_episodes = 5000,
            BIN = [1, 1, 1, 1, 1, 1, 1, 1]
        )

        no_of_runs = 0
        while(True):
            agent.solve()
            no_of_runs += 1

            if no_of_runs == 1:
                break

        point_dict_basic_agent_four, point_list_basic_agent_four, specifier_basic_agent_four = agent.OutputTrainingData('basic')
        point_dict_consecutive_agent_four, point_list_consecutive_agent_four, specifier_consecutive_agent_four = agent.OutputTrainingData('consecutive')


    if SAVE_AGENT_FOUR == True:
        with open('PointDictBasic_Graph_13_14_15_AGENT_FOUR.pickle', 'wb') as handle:
            pickle.dump(point_dict_basic_agent_four, handle, protocol=pickle.HIGHEST_PROTOCOL)
            handle.close()

        with open('PointListBasic_Graph_13_14_15_AGENT_FOUR.pickle', 'wb') as handle:
            pickle.dump(point_list_basic_agent_four, handle, protocol=pickle.HIGHEST_PROTOCOL)
            handle.close()

        with open('SpecifierBasic_Graph_13_14_15_AGENT_FOUR.pickle', 'wb') as handle:
            pickle.dump(specifier_basic_agent_four, handle, protocol=pickle.HIGHEST_PROTOCOL)
            handle.close()

        with open('PointDictConsecutive_Graph_13_14_15_AGENT_FOUR.pickle', 'wb') as handle:
            pickle.dump(point_dict_consecutive_agent_four, handle, protocol=pickle.HIGHEST_PROTOCOL)
            handle.close()

        with open('PointListConsecutive_Graph_13_14_15_AGENT_FOUR.pickle', 'wb') as handle:
            pickle.dump(point_list_consecutive_agent_four, handle, protocol=pickle.HIGHEST_PROTOCOL)
            handle.close()

        with open('SpecifierConsecutive_Graph_13_14_15_AGENT_FOUR.pickle', 'wb') as handle:
            pickle.dump(specifier_consecutive_agent_four, handle, protocol=pickle.HIGHEST_PROTOCOL)
            handle.close()

    if LOAD_AGENT_FOUR == True:
        try:
            with open('PointDictBasic_Graph_13_14_15_AGENT_FOUR.pickle', 'rb') as handle:
                point_dict_basic_agent_four = pickle.load(handle)
                handle.close()

            with open('PointListBasic_Graph_13_14_15_AGENT_FOUR.pickle', 'rb') as handle:
                point_list_basic_agent_four = pickle.load(handle)
                handle.close()

            with open('SpecifierBasic_Graph_13_14_15_AGENT_FOUR.pickle', 'rb') as handle:
                specifier_basic_agent_four = pickle.load(handle)
                handle.close()

            with open('PointDictConsecutive_Graph_13_14_15_AGENT_FOUR.pickle', 'rb') as handle:
                point_dict_consecutive_agent_four = pickle.load(handle)
                handle.close()

            with open('PointListConsecutive_Graph_13_14_15_AGENT_FOUR.pickle', 'rb') as handle:
                point_list_consecutive_agent_four = pickle.load(handle)
                handle.close()

            with open('SpecifierConsecutive_Graph_13_14_15_AGENT_FOUR.pickle', 'rb') as handle:
                specifier_consecutive_agent_four = pickle.load(handle)
                handle.close()
        except:
            print('No values loaded.')


    if RUN_AGENT_FIVE == True:
        agent = QLearningAgent(
            dumpPickle = True,
            pickle = 'QLearningTable.pickle',
            alpha = 1.0,
            gamma = 0.0,
            epsilon_min = 1.0,
            epsilon_max = 1.0,
            training_episodes = 5000,
            BIN = [20, 20, 1, 1, 1, 1, 1, 1]
        )

        no_of_runs = 0
        while(True):
            agent.solve()
            no_of_runs += 1

            if no_of_runs == 1:
                break

        point_dict_basic_agent_five, point_list_basic_agent_five, specifier_basic_agent_five = agent.OutputTrainingData('basic')
        point_dict_consecutive_agent_five, point_list_consecutive_agent_five, specifier_consecutive_agent_five = agent.OutputTrainingData('consecutive')


    if SAVE_AGENT_FIVE == True:
        with open('PointDictBasic_Graph_13_14_15_AGENT_FIVE.pickle', 'wb') as handle:
            pickle.dump(point_dict_basic_agent_five, handle, protocol=pickle.HIGHEST_PROTOCOL)
            handle.close()

        with open('PointListBasic_Graph_13_14_15_AGENT_FIVE.pickle', 'wb') as handle:
            pickle.dump(point_list_basic_agent_five, handle, protocol=pickle.HIGHEST_PROTOCOL)
            handle.close()

        with open('SpecifierBasic_Graph_13_14_15_AGENT_FIVE.pickle', 'wb') as handle:
            pickle.dump(specifier_basic_agent_five, handle, protocol=pickle.HIGHEST_PROTOCOL)
            handle.close()

        with open('PointDictConsecutive_Graph_13_14_15_AGENT_FIVE.pickle', 'wb') as handle:
            pickle.dump(point_dict_consecutive_agent_five, handle, protocol=pickle.HIGHEST_PROTOCOL)
            handle.close()

        with open('PointListConsecutive_Graph_13_14_15_AGENT_FIVE.pickle', 'wb') as handle:
            pickle.dump(point_list_consecutive_agent_five, handle, protocol=pickle.HIGHEST_PROTOCOL)
            handle.close()

        with open('SpecifierConsecutive_Graph_13_14_15_AGENT_FIVE.pickle', 'wb') as handle:
            pickle.dump(specifier_consecutive_agent_five, handle, protocol=pickle.HIGHEST_PROTOCOL)
            handle.close()

    if LOAD_AGENT_FIVE == True:
        try:
            with open('PointDictBasic_Graph_13_14_15_AGENT_FIVE.pickle', 'rb') as handle:
                point_dict_basic_agent_five = pickle.load(handle)
                handle.close()

            with open('PointListBasic_Graph_13_14_15_AGENT_FIVE.pickle', 'rb') as handle:
                point_list_basic_agent_five = pickle.load(handle)
                handle.close()

            with open('SpecifierBasic_Graph_13_14_15_AGENT_FIVE.pickle', 'rb') as handle:
                specifier_basic_agent_five = pickle.load(handle)
                handle.close()

            with open('PointDictConsecutive_Graph_13_14_15_AGENT_FIVE.pickle', 'rb') as handle:
                point_dict_consecutive_agent_five = pickle.load(handle)
                handle.close()

            with open('PointListConsecutive_Graph_13_14_15_AGENT_FIVE.pickle', 'rb') as handle:
                point_list_consecutive_agent_five = pickle.load(handle)
                handle.close()

            with open('SpecifierConsecutive_Graph_13_14_15_AGENT_FIVE.pickle', 'rb') as handle:
                specifier_consecutive_agent_five = pickle.load(handle)
                handle.close()
        except:
            print('No values loaded.')

    ## Graph: Effect of hyperparameters and discussion of the results.
    ## Hyperparameters: self.epsilon, self.gamma, self.alpha, binning values
    ## Conversely, a gamma of zero will cause the agent to only value immediate rewards, which only works with very detailed reward functions.
    if (RUN_AGENT_ONE == True or LOAD_AGENT_ONE == True) and (RUN_AGENT_TWO == True or LOAD_AGENT_TWO == True) and (RUN_AGENT_THREE == True or LOAD_AGENT_THREE == True) and (RUN_AGENT_FOUR == True or LOAD_AGENT_FOUR == True) and (RUN_AGENT_FIVE == True or LOAD_AGENT_FIVE == True):
        grapher_3 = Grapher(point_dict_basic, point_list_basic, specifier_basic)
        grapher_3.AddDictionaries([point_dict_basic_agent_two, point_dict_basic_agent_three, point_dict_basic_agent_four, point_dict_basic_agent_five])
        grapher_3.AddLists([point_list_basic_agent_two, point_list_basic_agent_three, point_list_basic_agent_four, point_list_basic_agent_five])
        grapher_3.Display_Graph(specifier_basic[0], specifier_basic[1], specifier_basic[2])

        grapher_4 = Grapher(point_dict_consecutive, point_list_consecutive, specifier_consecutive)
        grapher_4.AddDictionaries([point_dict_consecutive_agent_two, point_dict_consecutive_agent_three, point_dict_consecutive_agent_four, point_dict_consecutive_agent_five])
        grapher_4.AddLists([point_list_consecutive_agent_two, point_list_consecutive_agent_three, point_list_consecutive_agent_four, point_list_consecutive_agent_five])
        grapher_4.Display_Graph(specifier_consecutive[0], specifier_consecutive[1], specifier_consecutive[2])     

if __name__ == "__main__":
    main()