import pyomo.environ as pyo
from pyomo.opt import SolverFactory
import pandas as pd
import os 
from  itertools import product
from datetime import datetime, timedelta
import numpy as np
import cloudpickle
import time
from pyomo.core.util import quicksum, sum_product


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
        

    def __find_aircraft_class(self, flights_data):
        aircraft_class_dict = dict()
        aircraft_classes_data = self.get_aircraft_classes()
        for (flight_number, capacity_of_flight) in flights_data['flight_AC_PAX_capacity_total'].items():
            for (aircraft_class, number_seats) in aircraft_classes_data['Max_Seats'].items():
                if capacity_of_flight <= number_seats:
                    aircraft_class_dict.update({flight_number: aircraft_class })
                    break
        return aircraft_class_dict



    def get_flights(self):
        flights = super().get_flights()
        flights['flight_datetime'] = {i:j for (i,j) in enumerate(list(map(lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S'), list(flights['flight_datetime'].values()))))}
        flights['quantity_busses'] = {key: np.ceil(flights['flight_PAX'][key] / self.bus_capacity) for key in flights['flight_PAX'].keys()}
        flights['aircraft_class'] = self.__find_aircraft_class(flights)
        return flights

    def get_aircraft_classes(self):
        aircraft_classes = super().get_aircraft_classes()
        aircraft_classes = {'Max_Seats': {k: v for k, v in sorted(aircraft_classes['Max_Seats'].items(), key=lambda item: item[1])}}
        return aircraft_classes





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
        self.opt.options['ratioGap'] = 0.0000001
        self.opt.options['sec'] = 40000

    
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
        return quicksum([self.model.AS_occupied[flight, stand] *
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

    def busses_cost_func(self, model, flight):
        # При использовании удалённых МС ВС для посадки/высадки пассажиров необходимо использовать перронные автобусы. Вместимость одного перронного автобуса 80 пассажиров. Время движения автобуса от терминала и стоимость минуты использования автобуса указаны в соответствующих таблицах.
        return quicksum([self.model.AS_occupied[flight, stand] *
                    self.FLIGHTS_DATA['quantity_busses'][flight] *
                    self.AIRCRAFT_STANDS_DATA[str(self.FLIGHTS_DATA['flight_terminal_#'][flight])][stand] *
                    (1 - self.teletrap_can_be_used(flight, stand))
                    for stand in self.AIRCRAFT_STANDS])
        
    def time_calculate_func(self, flight, aircraft_stand, time):
        flight_time = self.FLIGHTS_DATA['flight_datetime'][flight]
        taxiing_time = int(self.AIRCRAFT_STANDS_DATA['Taxiing_Time'][aircraft_stand])
        arrival_or_depature = self.FLIGHTS_DATA['flight_AD'][flight]
        use_trap_flg = self.teletrap_can_be_used(flight, aircraft_stand)

        if use_trap_flg:
            column_handling_time = 'JetBridge_Handling_Time'
        else: 
            column_handling_time = 'Away_Handling_Time'
        aircraft_class = self.FLIGHTS_DATA['aircraft_class'][flight]
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
    
    def AS_using_cost_def(self, model, flight):
        # Стоимость использования MC VC
        return quicksum([self.model.AS_occupied[flight, stand] *
                    self.HANGLING_TIME['JetBridge_Handling_Time'][self.FLIGHTS_DATA['aircraft_class'][stand]] *
                    self.HANDLING_RATES_DATA['JetBridge_Aircraft_Stand_Cost_per_Minute'] * self.teletrap_can_be_used(flight, stand) +
                    self.model.AS_occupied[flight, stand] *
                    self.HANGLING_TIME['Away_Handling_Time'][self.FLIGHTS_DATA['aircraft_class'][stand]] *
                    self.HANDLING_RATES_DATA['Away_Aircraft_Stand_Cost_per_Minute'] * self.teletrap_can_be_used(flight, stand)
                    for stand in self.AIRCRAFT_STANDS])
    
    def only_one_flight_per_place_func(self, model, stand, time):
        return quicksum([self.time_calculate_func(flight, stand, time) * model.AS_occupied[flight, stand] for flight in self.FLIGHTS]) <= 1
    
    def teletrap_can_be_used_on_stand(self, stand):
        return self.AIRCRAFT_STANDS_DATA['JetBridge_on_Arrival'][stand] != 'N' and self.AIRCRAFT_STANDS_DATA['JetBridge_on_Departure'][stand] != 'N'
    
    def two_wide_near_are_prohibited_func(self, model, stand, time):

        if stand - 1 in self.AIRCRAFT_STANDS:
            left_stand = quicksum([self.time_calculate_func(flight, stand - 1, time) * model.AS_occupied[flight, stand - 1] for flight in self.FLIGHTS_WIDE])
        else:
            left_stand = 0
        
        middle_stand = quicksum([self.time_calculate_func(flight, stand, time) * model.AS_occupied[flight, stand] for flight in self.FLIGHTS_WIDE])

        if stand + 1 in self.AIRCRAFT_STANDS:
            right_stand = quicksum([self.time_calculate_func(flight, stand + 1, time) * model.AS_occupied[flight, stand + 1] for flight in self.FLIGHTS_WIDE])
        else:
            right_stand = 0
        
        out = (left_stand + middle_stand + right_stand) <= 1
        if isinstance(out, bool):
            return pyo.Constraint.Feasible
        else:
            return out
    
    def every_flight_must_have_its_stand_func(self, model, flight):
        return quicksum([self.model.AS_occupied[flight, stand] for stand in self.AIRCRAFT_STANDS]) == 1


    def make_model(self, start_dt=datetime(2019, 5, 17, 0, 0), end_dt=datetime(2019, 5, 17, 23, 55)):
        t = time.time()
        self.set_solver()

        self.model = pyo.ConcreteModel()
        # Рейсы
        self.FLIGHTS = self.FLIGHTS_DATA['index'].values()
        # Места стоянки
        self.AIRCRAFT_STANDS = self.AIRCRAFT_STANDS_DATA['index'].values()
        # Временные отрезки
        self.TIMES = self.__get_times(start_dt=start_dt, end_dt=end_dt)
        # Места стоянки с телетрапом
        self.AIRCRAFT_STANDS_WITH_TRAPS = [stand for stand in self.AIRCRAFT_STANDS if self.teletrap_can_be_used_on_stand(stand)]

        self.FLIGHTS_WIDE = [flight for flight in self.FLIGHTS if self.FLIGHTS_DATA['aircraft_class'][flight] == 'Wide_Body']

        new_t = time.time()
        print(t - new_t, "1")
        new_t2 = time.time()
        # занимаемые места (Рейс * МC) - переменные
        self.model.AS_occupied = pyo.Var(self.FLIGHTS, self.AIRCRAFT_STANDS, within=pyo.Binary, initialize=0)

        new_t = time.time()
        print(new_t - new_t2, "2")
        new_t2 = time.time()
        # Cтоимость руления по аэродрому
        self.model.airport_taxiing_cost = pyo.Expression(self.FLIGHTS, rule=self.airport_taxiing_cost_func)
        new_t = time.time()
        print(new_t - new_t2, "3")
        new_t2 = time.time()
        # Стоимость использования МС ВС
        self.model.AS_using_cost = pyo.Expression(self.FLIGHTS, rule=self.AS_using_cost_def)
        new_t = time.time()
        print(new_t - new_t2, "4")
        new_t2 = time.time()
        # Стоимость использования перронных автобусов для посадки/высадки пассажиров
        self.model.busses_cost = pyo.Expression(self.FLIGHTS, rule=self.busses_cost_func)
        new_t = time.time()
        print(new_t - new_t2, "5")
        new_t2 = time.time()
        # Целевая переменная
        self.model.OBJ = pyo.Objective(expr=quicksum([self.model.airport_taxiing_cost[flight] for flight in self.FLIGHTS]) +\
                                            quicksum([self.model.AS_using_cost[stand] for stand in self.AIRCRAFT_STANDS]) +\
                                            quicksum([self.model.busses_cost[flight] for flight in self.FLIGHTS]), sense=pyo.minimize)

        new_t = time.time()
        print(new_t - new_t2, "6")
        new_t2 = time.time()
        self.model.only_one_flight_per_place = pyo.Constraint(self.AIRCRAFT_STANDS, self.TIMES, rule=self.only_one_flight_per_place_func)
    
        new_t = time.time()
        print(new_t - new_t2, "7")
        new_t2 = time.time()

        self.model.two_wide_near_are_prohibited = pyo.Constraint(self.AIRCRAFT_STANDS_WITH_TRAPS, self.TIMES, rule=self.two_wide_near_are_prohibited_func)

        new_t = time.time()
        print(new_t - new_t2, "8")
        new_t2 = time.time()

        self.model.every_flight_must_have_its_stand = pyo.Constraint(self.FLIGHTS, rule=self.every_flight_must_have_its_stand_func)
        
        new_t = time.time()
        print(new_t - new_t2, "9")
        new_t2 = time.time()
        self.opt_output = self.opt.solve(self.model, logfile='SOLVE_LOG', solnfile='SOLNFILE')
        print(self.opt_output)

    def get_model(self):
        return self.model

    def get_solution(self):
        assert self.model is not None
        AS_occupied_data = pd.DataFrame().from_dict(self.model.AS_occupied.extract_values(), orient='index', columns=['busy'])
        AS_occupied_data['flight'] = AS_occupied_data.index.map(lambda x: x[0])
        AS_occupied_data['stand'] = AS_occupied_data.index.map(lambda x: x[1])
        AS_occupied_data = AS_occupied_data.reset_index(drop=True)
        return AS_occupied_data
        

if __name__ == "__main__":
    d = DataExtended()
    
    opt = OptimizeDay(d)
    opt.make_model(datetime(2019, 5, 17, 0, 0), datetime(2019, 5, 17, 23, 55))
    o = opt.get_model()


