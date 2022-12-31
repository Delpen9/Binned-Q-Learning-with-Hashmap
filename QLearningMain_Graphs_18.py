from QLearningAgent import QLearningAgent
from Grapher import Grapher
import numpy as np
import pickle

def main():
    ## ------------------
    ## CONFIG SETTINGS
    ## ------------------
    SAVE_AGENT_ONE = False

    RUN_AGENT_ONE = False

    LOAD_AGENT_ONE = True
    ## ------------------

    if RUN_AGENT_ONE == True:
        agent = QLearningAgent(
            dumpPickle = True,
            pickle = 'QLearningTable.pickle',
            alpha = 1.0,
            gamma = 0.0,
            epsilon_min = 0.0,
            epsilon_max = 0.0,
            training_episodes = 5000,
            BIN = [10, 10, 10, 10, 10, 10, 1, 1]
        )

        no_of_runs = 0
        while(True):
            agent.solve()
            no_of_runs += 1

            if no_of_runs == 8:
                break

        point_dict_basic, point_list_basic, specifier_basic = agent.OutputTrainingData('basic')
        point_dict_consecutive, point_list_consecutive, specifier_consecutive = agent.OutputTrainingData('consecutive')

    if SAVE_AGENT_ONE == True:
        with open('PointDictBasic_Graph_18.pickle', 'wb') as handle:
            pickle.dump(point_dict_basic, handle, protocol=pickle.HIGHEST_PROTOCOL)
            handle.close()

        with open('PointListBasic_Graph_18.pickle', 'wb') as handle:
            pickle.dump(point_list_basic, handle, protocol=pickle.HIGHEST_PROTOCOL)
            handle.close()

        with open('SpecifierBasic_Graph_18.pickle', 'wb') as handle:
            pickle.dump(specifier_basic, handle, protocol=pickle.HIGHEST_PROTOCOL)
            handle.close()

        with open('PointDictConsecutive_Graph_18.pickle', 'wb') as handle:
            pickle.dump(point_dict_consecutive, handle, protocol=pickle.HIGHEST_PROTOCOL)
            handle.close()

        with open('PointListConsecutive_Graph_18.pickle', 'wb') as handle:
            pickle.dump(point_list_consecutive, handle, protocol=pickle.HIGHEST_PROTOCOL)
            handle.close()

        with open('SpecifierConsecutive_Graph_18.pickle', 'wb') as handle:
            pickle.dump(specifier_consecutive, handle, protocol=pickle.HIGHEST_PROTOCOL)
            handle.close()

    if LOAD_AGENT_ONE == True:
        try:
            with open('PointDictBasic_Graph_18.pickle', 'rb') as handle:
                point_dict_basic = pickle.load(handle)
                handle.close()

            with open('PointListBasic_Graph_18.pickle', 'rb') as handle:
                point_list_basic = pickle.load(handle)
                handle.close()

            with open('SpecifierBasic_Graph_18.pickle', 'rb') as handle:
                specifier_basic = pickle.load(handle)
                handle.close()

            with open('PointDictConsecutive_Graph_18.pickle', 'rb') as handle:
                point_dict_consecutive = pickle.load(handle)
                handle.close()

            with open('PointListConsecutive_Graph_18.pickle', 'rb') as handle:
                point_list_consecutive = pickle.load(handle)
                handle.close()

            with open('SpecifierConsecutive_Graph_18.pickle', 'rb') as handle:
                specifier_consecutive = pickle.load(handle)
                handle.close()
        except:
            print('No values loaded.')

    grapher_1 = Grapher(point_dict_basic, point_list_basic, specifier_basic)
    grapher_1.Display_Graph(specifier_basic[0], specifier_basic[1], specifier_basic[2])
    
    grapher_2 = Grapher(point_dict_consecutive, point_list_consecutive, specifier_consecutive)
    grapher_2.Display_Graph(specifier_consecutive[0], specifier_consecutive[1], specifier_consecutive[2])

if __name__ == "__main__":
    main()