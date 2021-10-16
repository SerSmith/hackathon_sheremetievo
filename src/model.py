import pyomo.environ as pyo
from pyomo.opt import SolverFactory
import pandas as pd
import os 
from  itertools import product
from datetime import datetime, timedelta


class Data():
    def __init__(self, data_folder: str='../data'):
        self.data_folder = data_folder
        self.aircraft_classes_dict = None
        self.handling_rates_dict = None
        self.handling_time_dict = None
        self.aircraft_stands_dict = None
        self.flights_dict = None
    
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

    def get_aircraft_stands(self):
        if self.aircraft_stands_dict is None:
            aircraft_stands_folder = os.path.join(self.data_folder, 'Aircraft_Stands_Public.csv')
            aircraft_stands_pd = pd.read_csv(aircraft_stands_folder)
            self.aircraft_stands_dict  = aircraft_stands_pd.set_index('Aircraft_Stand').to_dict()
        return self.aircraft_stands_dict

    def get_flights(self):
        if self.flights_dict is None:
            flights_folder = os.path.join(self.data_folder, 'Timetable_Public.csv')
            flights_pd = pd.read_csv(flights_folder)
            flights_pd.drop(['Aircraft_Stand'], axis=1, inplace=True)
            self.flights_dict  = flights_pd.set_index('flight_number').to_dict()
        return self.flights_dict 




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
    
    @staticmethod
    def __get_times(start_dt, end_dt):
        result_5minutes_list = []
        current_dt = start_dt
        while current_dt<end_dt:
            result_5minutes_list.append(current_dt)
            current_dt = current_dt + timedelta(minutes = 5)
        return result_5minutes_list

        # Cтоимость руления по аэродрому
        def airport_taxiing_cost_func(self, flight):
            # Стоимость руления определяется как время руления (однозначно определяется МС ВС) умноженное на тариф за минуту руления
        return sum([self.model.AS_occupied[flight, stand, time] * AIRCRAFT_STANDS_DATA['Taxiing_Time'][stand] * HANDLING_RATES['Aircraft_Taxiing_Cost_per_Minute'] for stand, time in product(AIRCRAFT_STANDS, TIMES)]) 



    def make_model(self, start_dt = datetime.datetime(2019, 5, 17, 0, 0), end_dt = datetime.datetime(2019, 5, 17, 23, 55)):

        FLIGHTS_DATA = self.data.get_flights()
        AIRCRAFT_STANDS_DATA = self.data.get_aircraft_stands()
        HANDLING_RATES = self.data.get_handling_rates()


        # Рейсы
        FLIGHTS = FLIGHTS_DATA.keys()
        # Места стоянки
        AIRCRAFT_STANDS = AIRCRAFT_STANDS.keys()
        # Временные отрезки
        TIMES = self.__get_times(start_dt = start_dt, end_dt = end_dt)


        self.model = pyo.ConcreteModel()
    
        # занимаемые места (Рейс * МС * 5минутки) - переменные
        self.model.AS_occupied = pyo.Var(FLIGHTS, AIRCRAFT_STANDS, TIMES, within=pyo.Binary, initialize=0)

        # Cтоимость руления по аэродрому
        self.model.airport_taxiing_cost = pyo.Expression(FLIGHTS, rule=self.airport_taxiing_cost_func)


        # Cтоимость руления по аэродрому
        self.model.airport_taxiing_cost = pyo.Expression(FLIGHTS, rule=self.airport_taxiing_cost_func)
        
        # MC_VC:
        #     Стоимость
        #     Наличие трапа
        #     Возможность использования трапа
        #     время движения от терминала (1, 2, 3, 4, 5)



        # сущности:



        # смтоимость использования МС ВС
        # Стоимость использования перронных автобусов для посадки/высадки пассажировa


        pass

    def get_pyomo_obj(self):
        pass


if __name__ == "__main__":
    d = Data()
    opt = OptimizeDay(d)
    opt.make_model()


# Все места стоянок делятся на контактные (посадка/высадка через телетрап) и удалённые (посадка/высадка 
# с помощью перронного автобуса). Телетрап на данном МС доступен только в случае, если:

#     1. терминал рейса соответствует терминалу М
#     2. значение поля flight_ID рейса (метка МВЛ/ВВЛ – Domestic/Intern£tion£l) совпадает с соответствующей меткой поля
# JetBridge_on_Arriv(для прилетающих рейсов) или JetBridge_on_Dep£rture (для вылетающих рейсов) МС

# На МС с телетрапами существует дополнительное ограничение по расстановке ВС: на соседних МС
# (т.е. тех МС, у которых номер отличается на 1) не могут находиться одновременно два
# широкофюзеляжных ВС (ВС класса “Wide_Body”)

# При использовании удалённых МС ВС для посадки/высадки пассажиров необходимо использовать
# перронные автобусы. Вместимость одного перронного автобуса 80 пассажиров. Время движения
# автобуса от терминала и стоимость минуты использования автобуса указаны в соответствующих
# таблицах.

# Каждый тип ВС имеет свой протокол обслуживания (время обслуживания) на прилёт и вылет 
# (и, как следствие, разное время обслуживания и себестоимость)


# Тариф на использование МС ВС определяется наличием/отсутствием телетрапа на МС ВС (NB!
# Наличие телетрапа не означает возможность его использования, см. п. 1) и временем нахождения
# ВС на МС.