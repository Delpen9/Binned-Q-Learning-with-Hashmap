import numpy as np
import matplotlib.pyplot as plt

class Grapher(object):
    def __init__(self, point_dict, point_list, dict_value_specifier):
        self.point_dict = point_dict
        self.point_list = np.array(point_list)
        self.dict_value_specifier = dict_value_specifier

    def Display_Graph(self, y_axis, x_axis, tracks):
        plt.style.use('seaborn-whitegrid')
        color_list = ['blue', 'green', 'red', 'black', 'cyan', 'magenta', 'yellow', 'white']

        track_values = np.unique(self.point_dict.get(tracks))
        for track_value, color_value in zip(track_values, color_list[:len(track_values)]):
            track_point_list = np.array([value for value in self.point_list if value[2] == track_value])
            track_point_list_dict = {self.dict_value_specifier[0] : track_point_list[:, 0], self.dict_value_specifier[1] : track_point_list[:, 1], self.dict_value_specifier[2] : track_point_list[:, 2]}

            data_length = np.arange(1, len(track_point_list) + 1)
            
            x_axis_values = track_point_list_dict.get(x_axis)
            x_axis_values = np.array([float(x_value) for x_value, i in zip(x_axis_values, data_length) if i % 100 == 0])
            
            y_axis_values = track_point_list_dict.get(y_axis)
            y_axis_values = np.array([float(y_value) for y_value, i in zip(y_axis_values, data_length) if i % 100 == 0])

            plt.plot(x_axis_values, y_axis_values, '-ok', color = color_value)

        plt.xlabel(x_axis)
        plt.ylabel(y_axis)
        plt.legend([str(track_value) for track_value in track_values])
        plt.show()

    def AddDictionaries(self, dictionary_list):
        for dictionary in dictionary_list:
            for label in self.dict_value_specifier:
                temp_index_value = np.array(self.point_dict.get(label))
                temp_index_value = np.concatenate((temp_index_value, np.array(dictionary.get(label))))
                self.point_dict.update({label:temp_index_value})

    def AddLists(self, lists):
        for list_value in lists:
            self.point_list = np.concatenate((self.point_list, np.array(list_value)))