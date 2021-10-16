import pyomo.environ as pyo
from pyomo.opt import SolverFactory
import pandas as pd
import os 


class Data():
    def __init__(self, data_folder: str='../data'):
        self.data_folder = data_folder
        self.aircraft_classes_dict = None
        self.handling_rates_dict = None
        self.handling_time_dict = None
    
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

    def get_handling_time(self):
        if self.handling_time_dict is None:
            handling_time_folder = os.path.join(self.data_folder, 'Handling_Time_Public.csv')
            handling_time_pd = pd.read_csv(handling_time_folder)
            handling_time_pd = handling_time_pd.rename(columns={'JetBridge_Handling_Time': 'JetBridge',
        	                                                    'Away_Handling_Time': 'Away'})
            self.handling_time_dict = handling_time_pd.set_index('Aircraft_Class').to_dict()
        return self.handling_time_dict



class OptimizationSolution():
    def __init__(self):
        pass

    def check_solution(self):
        pass

    def get_solution_file(self):
        pass   

class OptimizeDay:
    def __init__(self, data: Data):
        self.model = None
        self.data = data
        pass
    
    def get_times(self):
        return 

    def make_model(self):

        FLIGHTS_DATA = self.data.get_flights()
        AIRCRAFT_STANDS_DATA = self.data.get_aircraft_stands()

        # Рейсы
        FLIGHTS = FLIGHTS_DATA.keys()
        # Места стоянки
        AIRCRAFT_STANDS = AIRCRAFT_STANDS.keys()
        # Временные отрезки
        TIMES = self.get_times()


        self.model = pyo.ConcreteModel()
    
        # занимаемые места (Рейс * МС * 5минутки) - переменные
        self.model.stops = pyo.Var(FLIGHTS, visuals, within=pyo.Binary, initialize=0)
        # MC_VC:
        #     Стоимость
        #     Наличие трапа
        #     Возможность использования трапа
        #     время движения от терминала (1, 2, 3, 4, 5)



        # сущности:

        # МС (количество МС)
        # 5минутка (720)


        # стоимость руления по аэродрому (количество рейсов)
        # смтоимость использования МС ВС
        # Стоимость использования перронных автобусов для посадки/высадки пассажировa


        pass

    def get_pyomo_obj(self):
        pass


if __name__ == "__main__":
    d = Data()
    opt = OptimizeDay(d)
    opt.make_model()




