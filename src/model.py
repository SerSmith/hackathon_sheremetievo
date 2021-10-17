import pyomo.environ as pyo
from pyomo.opt import SolverFactory
import pandas as pd
import os 
from  itertools import product
from datetime import datetime, timedelta
import numpy as np
import pickle

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
            self.handling_rates_dict = handling_rates_pd.set_index('Name').to_dict()['Value']
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
            aircraft_stands_pd = aircraft_stands_pd.set_index('Aircraft_Stand')
            aircraft_stands_pd['index'] = aircraft_stands_pd.index
            self.aircraft_stands_dict = aircraft_stands_pd.to_dict()
        return self.aircraft_stands_dict

    def get_flights(self):
        if self.flights_dict is None:
            flights_folder = os.path.join(self.data_folder, 'Timetable_Public.csv')
            flights_pd = pd.read_csv(flights_folder)
            flights_pd = flights_pd.reset_index(drop=True)
            flights_pd['index'] = flights_pd.index
            flights_pd.drop(['Aircraft_Stand'], axis=1, inplace=True)
            self.flights_dict  = flights_pd.to_dict()
        return self.flights_dict 


class DataExtended(Data):
    def __init__(self, data_folder: str='../data', bus_capacity:int = 80 ):
        super().__init__(data_folder)
        self.bus_capacity = bus_capacity
        

    def get_flights(self):
        flights = super().get_flights()
        flights['quantity_busses'] = {key: np.ceil(flights['flight_PAX'][key] / self.bus_capacity) for key in flights['flight_PAX'].keys()}
        return flights





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
        self.FLIGHTS_DATA = data.get_flights()
        self.AIRCRAFT_STANDS_DATA = data.get_aircraft_stands()
        self.HANDLING_RATES_DATA = data.get_handling_rates()
        self.AIRCRAFT_CLASSES_DATA = data.get_aircraft_classes()
        self.HANGLING_TIME = data.get_handling_time()

        self.FLIGHTS = None
        self.AIRCRAFT_STANDS = None
        self.TIMES = None

        self.opt = None

    def set_solver(self):
        self.opt = SolverFactory('cbc', executable="/usr/local/bin/cbc")
        self.opt.options['ratioGap'] = 0.1
        self.opt.options['sec'] = 30

    
    @staticmethod
    def __get_times(start_dt, end_dt):
        result_5minutes_list = []
        current_dt = start_dt
        while current_dt<end_dt:
            result_5minutes_list.append(current_dt)
            current_dt = current_dt + timedelta(minutes = 5)
        return result_5minutes_list

    # Cтоимость руления по аэродрому
    def airport_taxiing_cost_func(self, model, flight):
        # Стоимость руления определяется как время руления (однозначно определяется МС ВС) умноженное на тариф за минуту руления
        # TODO учесть, что некорректно может считаться из-замножественного time
        return sum([self.model.AS_occupied[flight, stand] *
                    self.AIRCRAFT_STANDS_DATA['Taxiing_Time'][stand] *
                    self.HANDLING_RATES_DATA['Aircraft_Taxiing_Cost_per_Minute']
                    for stand in self.AIRCRAFT_STANDS])
    

    def teletrap_can_be_used(self, flight, stand):
        """ Телетрап на данном МС доступен только в случае, если:
        1)терминал рейса соответствует терминалу МC
        2)значение поля flight_ID рейса (метка МВЛ/ВВЛ – Domestic/Intern£tion£l) совпадает с соответствующей меткой поля JetBridge_on_Arriv£l (для прилетающих рейсов) или JetBridge_on_Dep£rture (для вылетающих рейсов) МС
        """
        # TODO что делать если терминал пропущен
        # TODO проверить На МС с телетрапами существует дополнительное ограничение по расстановке ВС: на соседних МС (т.е. тех МС, у которых номер отличается на 1) не могут находиться одновременно два широкофюзеляжных ВС (ВС класса “Wide_Body”)
        cond1 = self.FLIGHTS_DATA['flight_terminal_#'][flight] == self.AIRCRAFT_STANDS_DATA['Terminal'][stand]
        cond2 = ((self.FLIGHTS_DATA['flight_ID'][flight] == self.AIRCRAFT_STANDS_DATA['JetBridge_on_Arrival'][stand])\
                 and (self.FLIGHTS_DATA['flight_AD'][flight] == 'A'))\
                 or\
                ((self.FLIGHTS_DATA['flight_ID'][flight] == self.AIRCRAFT_STANDS_DATA['JetBridge_on_Departure'][stand]) and\
                (self.FLIGHTS_DATA['flight_AD'][flight] == 'D'))

        return cond1 and cond2
    
    def find_aircraft_class(self, flight):
        """ Находим тип ВС ('Regional', 'Narrow_Body', 'Wide_Body') """
        capacity_of_flight = self.FLIGHTS_DATA['flight_AC_PAX_capacity_total'][flight]
        # Если будут другие данные, то надо добавить сортировку!
        for (aircraft_class, number_seats) in self.AIRCRAFT_CLASSES_DATA['Max_Seats'].items():
            if capacity_of_flight <= number_seats:
                return aircraft_class

    def busses_cost_func(self, model, flight):
        # При использовании удалённых МС ВС для посадки/высадки пассажиров необходимо использовать перронные автобусы. Вместимость одного перронного автобуса 80 пассажиров. Время движения автобуса от терминала и стоимость минуты использования автобуса указаны в соответствующих таблицах.
        return sum([self.model.AS_occupied[flight, stand] *
                    self.FLIGHTS_DATA['quantity_busses'][flight] *
                    self.AIRCRAFT_STANDS_DATA[str(self.FLIGHTS_DATA['flight_terminal_#'][flight])][stand] *
                    (1 - self.teletrap_can_be_used(flight, stand))
                    for stand in self.AIRCRAFT_STANDS])
        
    def time_calculate_func(self, model, flight, aircraft_stand, time):
        flight_time = datetime.strptime(self.FLIGHTS_DATA['flight_datetime'][flight], '%Y-%m-%d %H:%M:%S')
        taxiing_time = int(self.AIRCRAFT_STANDS_DATA['Taxiing_Time'][aircraft_stand])
        arrival_or_depature = self.FLIGHTS_DATA['flight_AD'][flight]
        use_trap_flg = self.teletrap_can_be_used(flight, aircraft_stand)

        if use_trap_flg:
            column_handling_time = 'JetBridge'
        else: 
            column_handling_time = 'Away'
        aircraft_class = self.find_aircraft_class(flight)
        handling_time = self.HANGLING_TIME[column_handling_time][aircraft_class]

        if arrival_or_depature == 'D':
            if (flight_time - timedelta(minutes=taxiing_time) > time) & \
                (flight_time - timedelta(minutes=handling_time) - timedelta(minutes=taxiing_time) < time):
                    result = 1
            else:
                result = 0
        elif arrival_or_depature == 'A':
            if (flight_time + timedelta(minutes=taxiing_time) < time) & \
                (flight_time + timedelta(minutes=handling_time) + timedelta(minutes=taxiing_time) > time):
                    result = 1
            else:
                result = 0
        else:
            raise ValueError(f"arrival_or_depature имеет некорректное значение: {arrival_or_depature} , а должно быть A или D")

        return result
    
    def AS_using_cost_def(self, stand):
        return 0
    
    def only_one_flight_per_place(self, model, stand, time):
        return sum([model.AS_occupied_time[flight, stand, time] for flight in self.FLIGHTS] <= 1)
    
    # def two_wide_near_are_prohibited_func():
    #     return 0 


    def make_model(self, start_dt=datetime(2019, 5, 17, 0, 0), end_dt=datetime(2019, 5, 17, 23, 55)):

        self.set_solver()

        self.model = pyo.ConcreteModel()
        # Рейсы
        self.FLIGHTS = self.FLIGHTS_DATA['index'].values()
        # Места стоянки
        self.AIRCRAFT_STANDS = self.AIRCRAFT_STANDS_DATA['index'].values()
        # Временные отрезки
        self.TIMES = self.__get_times(start_dt=start_dt, end_dt=end_dt)

    
        # занимаемые места (Рейс * МC) - переменные
        self.model.AS_occupied = pyo.Var(self.FLIGHTS, self.AIRCRAFT_STANDS, within=pyo.Binary, initialize=0)

        # занимаемые времена с учетом времени
        print(len(self.FLIGHTS), len(self.AIRCRAFT_STANDS), len(self.TIMES))
        self.model.AS_occupied_time = pyo.Expression(self.FLIGHTS, self.AIRCRAFT_STANDS, self.TIMES, rule=lambda model, flight, stand, time: 0)
        print('hehe I am here')
        # Cтоимость руления по аэродрому
        self.model.airport_taxiing_cost = pyo.Expression(self.FLIGHTS, rule=self.airport_taxiing_cost_func)

        # Стоимость использования МС ВС
        self.model.AS_using_cost = pyo.Expression(self.AIRCRAFT_STANDS, rule=self.AS_using_cost_def)

        # Стоимость использования перронных автобусов для посадки/высадки пассажиров
        self.model.busses_cost = pyo.Expression(self.FLIGHTS, rule=self.busses_cost_func)

        # Целевая переменная
        self.model.OBJ = pyo.Objective(expr=sum([self.model.airport_taxiing_cost[flight] for flight in self.FLIGHTS]) +\
                                            sum([self.model.AS_using_cost[stand] for stand in self.AIRCRAFT_STANDS]) +\
                                            sum([self.model.busses_cost[flight] for flight in self.FLIGHTS]), sense=pyo.minimize)
        # self.model.OBJ = pyo.Objective(expr=0, sense=pyo.minimize)

        self.model.only_one_flight_per_place = pyo.Constraint(self.AIRCRAFT_STANDS, self.TIMES, rule=self.only_one_flight_per_place_func)

        # self.model.two_wide_near_are_prohibited = pyo.Constraint(self.flights, self.TIMES, rule=self.two_wide_near_are_prohibited_func)

        self.opt_output = self.opt.solve(self.model, logfile='SOLVE_LOG', solnfile='SOLNFILE')


    def get_pyomo_obj(self):
        return self.model

if __name__ == "__main__":
    d = DataExtended()
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