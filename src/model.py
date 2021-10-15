# import pyomo.environ as pyo
# from pyomo.opt import SolverFactory
import pandas as pd
import os 

class Data():
    def __init__(self, data_folder: str='../data'):
        self.data_folder = data_folder
        self.aircraft_classes_dict = None
        self.handling_rates_dict = None
    
    def get_aircraft_classes(self):
        if self.aircraft_classes_dict is None:
            aircraft_folder = os.path.join(self.data_folder, 'AirCraftClasses_Public.csv')
            aircraft_pd = pd.read_csv(aircraft_folder)
            self.aircraft_classes_dict = aircraft_pd.set_index('Aircraft_Class').to_dict()
        
        return self.aircraft_classes_dict
    
    def get_handling_rates(self):
        if self.handling_rates_dict is None:
            handling_rates_folder = os.path.join(self.data_folder, 'Handling_Rates_Public.csv')
            handling_rates_pd = pd.read_csv(handling_rates_folder)
            self.handling_rates_dict = handling_rates_pd.set_index('Name').to_dict()
        return self.handling_rates_dict




class OptimizationSolution():
    def __init__(self):
        pass

    def check_solution(self):
        pass

    def get_solution_file(self):
        pass   

class OptimizeDay:
    def __init__(self):
        pass

    def make_model(self):
        pass



if __name__ == "__main__":
    d = Data()

    d.get_handling_rates()




MC_VC:
    Стоимость
    Наличие трапа
    Возможность использования трапа
    время движения от терминала (1, 2, 3, 4, 5)



сущности:

рейс (количество рейсов)
МС (количество МС)
5минутка (720)
занимаемые места (Рейс * МС * 5минутки) - переменные

стоимость руления по аэродрому (количество рейсов)
смтоимость использования МС ВС
Стоимость использования перронных автобусов для посадки/высадки пассажиров


