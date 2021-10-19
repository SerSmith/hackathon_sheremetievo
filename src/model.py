import pyomo.environ as pyo
from pyomo.opt import SolverFactory
import pandas as pd
import os 
from  itertools import product
from datetime import datetime, timedelta
import numpy as np
import cloudpickle
import time


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


    def __init__(self, data_folder: str='../data'):
        self.data_folder = data_folder
        self.aircraft_classes_df = None
        self.timetable_df = None
        self.aircraft_stands_df = None
        self.handling_rates_df = None
        self.handling_time_df = None
        self.RANDOM_STATE = 33
        self.BUS_CAPACITY = 80
        self.MIN_PARKING_DELTA = pd.Timedelta('5 minutes')

    
    def load_all_data(self):
        aircraft_classes_path = os.path.join(self.data_folder, 'AirCraftClasses_Public.csv')
        self.aircraft_classes_df = pd.read_csv(aircraft_classes_path)

        timetable_path = os.path.join(self.data_folder, 'Timetable_Public.csv')
        self.timetable_df = pd.read_csv(timetable_path)

        aircraft_stands_path = os.path.join(self.data_folder, 'Aircraft_Stands_Public.csv')
        self.aircraft_stands_df = pd.read_csv(aircraft_stands_path)

        handling_rates_path = os.path.join(self.data_folder, 'Handling_Rates_Public.csv')
        self.handling_rates_df = pd.read_csv(handling_rates_path)

        handling_time_path = os.path.join(self.data_folder, 'Handling_Time_Public.csv')
        self.handling_time_df = pd.read_csv(handling_time_path)


    def determine_aircraft_classes(self):
        capacity_class_map = \
            self.aircraft_classes_df.set_index('Max_Seats')['Aircraft_Class'].to_dict()
        capacities = sorted(capacity_class_map.keys())
        aircraft_class = self.timetable_df['flight_AC_PAX_capacity_total'].copy()
        aircraft_class[aircraft_class <= capacities[0]] = 0
        aircraft_class[(aircraft_class > capacities[0]) & (aircraft_class <= capacities[1])] = 1
        aircraft_class[(aircraft_class > capacities[1]) & (aircraft_class <= capacities[2])] = 2
        replace_dict = {i: capacity_class_map[capacities[i]] for i in range(len(capacities))}
        return aircraft_class.replace(replace_dict).reset_index(drop=True)

    @staticmethod
    def get_parking_deltas(row):
        start_points = row['start_parking']
        end_points = row['end_parking']
        deltas_list = []
        if len(start_points) > 1:
            for i in range(1, len(start_points)):
                parking_delta = start_points[i] - end_points[i-1]
                deltas_list.append(parking_delta)
        return deltas_list


    def check_solution(self, solution_df=None):

        self.load_all_data()
        aircraft_class = self.determine_aircraft_classes()
        handling_rates_dict = self.handling_rates_df.set_index('Name').to_dict()['Value']

        if solution_df is None:
            aircraft_stand = self.aircraft_stands_df['Aircraft_Stand'].\
                sample(n=aircraft_class.shape[0],
                       replace=True,
                       random_state=self.RANDOM_STATE).reset_index(drop=True)

        flight_data = self.timetable_df.copy()
        flight_data['Aircraft_Class'] = aircraft_class
        flight_data['Aircraft_Stand'] = aircraft_stand
        flight_data = flight_data.merge(self.aircraft_stands_df, on='Aircraft_Stand')
        flight_data = flight_data.merge(self.handling_time_df, on='Aircraft_Class')
        flight_data['flight_datetime'] = pd.to_datetime(flight_data['flight_datetime'])

        # считаем, сколько от МС ехать на автобусе до обозначенного терминала прилёта/вылета
        flight_data['Bus_Time'] = flight_data.apply(
            lambda x: x[str(x['flight_terminal_#'])], 
            axis=1
            )

        # считаем, сколько надо автобусов, чтобы перевезти пассажиров
        flight_data['num_needed_Buses'] = (
            flight_data['flight_PAX'] // self.BUS_CAPACITY
            +
            (flight_data['flight_PAX'] % self.BUS_CAPACITY > 0)
        )

        # определяем, содержит ли выбранное для ВС МС телетрап
        flight_data['is_JetBridge'] = (
            (flight_data['JetBridge_on_Arrival'] != 'N')
            |
            (flight_data['JetBridge_on_Departure'] != 'N')
        )
            
        # соответствует ли терминал МС с телетрапом терминалу рейса
        flight_data['is_correct_JetBridge_Terminal'] = np.nan
        flight_data.loc[flight_data['is_JetBridge'], 'is_correct_JetBridge_Terminal'] = (
            flight_data.loc[flight_data['is_JetBridge'], 'Terminal']
            ==
            flight_data.loc[flight_data['is_JetBridge'], 'flight_terminal_#']
        )

        # соответствует ли тип JetBridge типу flight_ID (Внутренний/Международный рейс)
        flight_data['is_correct_JetBridge'] = np.nan
        arrival_jetbridge_cond = (flight_data['flight_AD'] == 'A') & flight_data['is_JetBridge']
        departure_jetbridge_cond = (flight_data['flight_AD'] == 'D') & flight_data['is_JetBridge']

        flight_data.loc[arrival_jetbridge_cond, 'is_correct_JetBridge'] = (
            flight_data.loc[arrival_jetbridge_cond, 'flight_ID']
            ==
            flight_data.loc[arrival_jetbridge_cond, 'JetBridge_on_Arrival']
        )
            
        flight_data.loc[departure_jetbridge_cond, 'is_correct_JetBridge'] = (
            flight_data.loc[departure_jetbridge_cond, 'flight_ID']
            ==
            flight_data.loc[departure_jetbridge_cond, 'JetBridge_on_Departure']
        )

        # считаем время обслуживания ВС на МС
        flight_data['Handling_Time'] = flight_data['Away_Handling_Time']
        # условие, что ВС стоит на МС с телетрапом у терминала, соответствующего терминалу прилета
        # и тип телетрапа соответствует типу рейса
        # в противном случае обслуживание осуществляется с помощью автобуса
        jet_bridge_can_be_used = (
            flight_data['is_JetBridge']
            &
            flight_data['is_correct_JetBridge_Terminal']
            &
            flight_data['is_correct_JetBridge']
        )
        flight_data.loc[jet_bridge_can_be_used, 'Handling_Time'] = \
            flight_data.loc[jet_bridge_can_be_used, 'JetBridge_Handling_Time']

        flight_data['JetBridge_can_be_used'] = jet_bridge_can_be_used

        # определяем время прибыия ВС на МС и время убытия
        arrival_cond = flight_data['flight_AD'] ==  'A'
        departure_cond = flight_data['flight_AD'] ==  'D'

        flight_data['start_parking'] = flight_data['flight_datetime']
        flight_data['end_parking'] = flight_data['flight_datetime']

        taxiing_time = pd.to_timedelta(flight_data['Taxiing_Time'], unit='minutes')
        handling_time = pd.to_timedelta(flight_data['Handling_Time'], unit='minutes')

        flight_data.loc[arrival_cond, 'start_parking'] += taxiing_time[arrival_cond]
        flight_data.loc[arrival_cond, 'end_parking'] = flight_data.loc[arrival_cond, 'start_parking'] + handling_time[arrival_cond]

        flight_data.loc[departure_cond, 'end_parking'] -= taxiing_time[departure_cond]
        flight_data.loc[departure_cond, 'start_parking'] = flight_data.loc[departure_cond, 'end_parking'] - handling_time[departure_cond]

        # определяем, не пересекаются ли ВС на МС по времени
        parking_deltas_df = \
            flight_data[['Aircraft_Stand', 'start_parking', 'end_parking']]\
                .copy().\
                    reset_index().\
                        sort_values(by='start_parking')
        parking_deltas_df = parking_deltas_df.groupby('Aircraft_Stand').agg(list).reset_index()

        parking_deltas_df['parking_deltas'] = parking_deltas_df.\
            apply(self.get_parking_deltas, axis=1)

        parking_deltas_df['is_parking_conflict'] = parking_deltas_df['parking_deltas'].apply(
            lambda x: any([True for delta in x if delta < self.MIN_PARKING_DELTA])
        )

        # получаем индексы всех конфликтующих стоянок
        parking_deltas_df['conflict_index'] = parking_deltas_df.apply(
            lambda x: sorted(
                set(
                    np.array(
                        [[x['index'][i], x['index'][i+1]] for i, delta in enumerate(x['parking_deltas']) if delta < self.MIN_PARKING_DELTA]
                        ).flatten(),
                )
            ),
            axis=1
        )
        # для всех рейсов, вовлеченных в конфликт, ставим соответствующий флаг
        parking_conflicts = parking_deltas_df[['is_parking_conflict', 'conflict_index']].\
            explode(column='conflict_index').dropna()
        parking_conflicts = parking_conflicts.set_index('conflict_index')
        parking_conflicts.index.name = None

        flight_data = flight_data.join(parking_conflicts)
        flight_data.loc[flight_data['is_parking_conflict'].isna(), 'is_parking_conflict'] = False


        # кейс на соседних МС с телетрапом стоят Wide_body

        wide_body_stands_cols = ['Aircraft_Stand', 'start_parking', 'end_parking']

        wide_body_stands = flight_data.loc[
            flight_data['is_JetBridge'] & (flight_data['Aircraft_Class'] == 'Wide_Body'), 
            wide_body_stands_cols
            ].sort_values(by=['Aircraft_Stand', 'start_parking']).reset_index()

        wide_body_stands['start_parking'] = wide_body_stands.apply(lambda x: [x['start_parking'], x['index']], axis=1)
        wide_body_stands['end_parking'] = wide_body_stands.apply(lambda x: [x['end_parking'], x['index']], axis=1)

        wide_body_stands = wide_body_stands.groupby('Aircraft_Stand').agg(list).reset_index()

        wide_body_stands['to_next_stand_distance'] = abs(wide_body_stands['Aircraft_Stand'].diff(-1))

        stands = wide_body_stands['Aircraft_Stand'].values
        to_next_distance = wide_body_stands['to_next_stand_distance'].values

        adjacent_stands = [[st, st+1] for st, dist in zip(stands, to_next_distance) if dist == 1]

        wide_body_stands['parking_points'] = wide_body_stands['start_parking'] + wide_body_stands['end_parking']
        wide_body_stands['parking_points'] = wide_body_stands['parking_points'].apply(sorted)
        wide_body_stands['parking_points'] = wide_body_stands.apply(
            lambda x: [point + [x['Aircraft_Stand']] for point in x['parking_points']],
            axis=1
            )

        stand_parking_map = wide_body_stands[['Aircraft_Stand', 'parking_points']].\
            set_index('Aircraft_Stand').to_dict()['parking_points']
        parking_intervals = [sorted(stand_parking_map[s[0]]+stand_parking_map[s[1]]) for s in adjacent_stands]
        problems_list = []
        index_list = []
        for adjacent in parking_intervals:
            for i in range(0, len(adjacent), 2):
                problems_list.append(adjacent[i][1] != adjacent[i+1][1])
                index_list.append([adjacent[i][1], adjacent[i+1][1]])

        wide_body_intersect = pd.DataFrame()
        wide_body_intersect['is_wide_body_intersect'] = problems_list
        wide_body_intersect['flight_index'] = index_list
        wide_body_intersect = wide_body_intersect.explode(column='flight_index').drop_duplicates().set_index('flight_index')
        wide_body_intersect.index.name = None

        flight_data = flight_data.join(wide_body_intersect)

        # расчёт целевых переменных
        # Стоимость руления по аэродрому
        flight_data['Taxiing_cost'] = flight_data['Taxiing_Time']*handling_rates_dict['Aircraft_Taxiing_Cost_per_Minute']

        # Стоимость места стоянки
        flight_data['Parking_usage_cost'] = handling_rates_dict['Away_Aircraft_Stand_Cost_per_Minute']
        flight_data.loc[flight_data['is_JetBridge'], 'Parking_usage_cost'] = handling_rates_dict['JetBridge_Aircraft_Stand_Cost_per_Minute']
        flight_data['Parking_usage_cost'] = flight_data['Parking_usage_cost'] * flight_data['Handling_Time']

        # Стоимость использования перронных автобусов для посадки/высадки пассажиров
        flight_data['Bus_usage_cost'] = flight_data['Bus_Time'] * flight_data['num_needed_Buses'] * handling_rates_dict['Bus_Cost_per_Minute']
        flight_data['Total_cost_Bus'] = flight_data['Taxiing_cost']	+ \
            flight_data['Parking_usage_cost'] + \
                flight_data['Bus_usage_cost']

        flight_data['Total_cost_Jetbridge_(if_possibe)'] = flight_data['Taxiing_cost']	+ \
            flight_data['Parking_usage_cost'] + \
                flight_data['Bus_usage_cost'] * ~flight_data['JetBridge_can_be_used']

        flight_data['is_wide_body_intersect'] = flight_data['is_wide_body_intersect'].fillna(False)
        
        return flight_data


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
        self.opt.options['ratioGap'] = 0.00000001
        self.opt.options['sec'] = 20000

    
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

    def busses_cost_func(self, model, flight):
        # При использовании удалённых МС ВС для посадки/высадки пассажиров необходимо использовать перронные автобусы. Вместимость одного перронного автобуса 80 пассажиров. Время движения автобуса от терминала и стоимость минуты использования автобуса указаны в соответствующих таблицах.
        return sum([self.model.AS_occupied[flight, stand] *
                    self.FLIGHTS_DATA['quantity_busses'][flight] *
                    self.AIRCRAFT_STANDS_DATA[str(self.FLIGHTS_DATA['flight_terminal_#'][flight])][stand] *
                    (1 - self.teletrap_can_be_used(flight, stand))
                    for stand in self.AIRCRAFT_STANDS])
        
    def time_calculate_func(self, model, flight, aircraft_stand, time):
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
        return result * self.model.AS_occupied[flight, aircraft_stand]
    
    def AS_using_cost_def(self, model, flight):
        # Стоимость использования MC VC
        return sum([self.model.AS_occupied[flight, stand] *
                    self.HANGLING_TIME['JetBridge_Handling_Time'][self.FLIGHTS_DATA['aircraft_class'][stand]] *
                    self.HANDLING_RATES_DATA['JetBridge_Aircraft_Stand_Cost_per_Minute'] * self.teletrap_can_be_used(flight, stand) +
                    self.model.AS_occupied[flight, stand] *
                    self.HANGLING_TIME['Away_Handling_Time'][self.FLIGHTS_DATA['aircraft_class'][stand]] *
                    self.HANDLING_RATES_DATA['Away_Aircraft_Stand_Cost_per_Minute'] * self.teletrap_can_be_used(flight, stand)
                    for stand in self.AIRCRAFT_STANDS])
    
    def only_one_flight_per_place_func(self, model, stand, time):
        return sum([model.AS_occupied_time[flight, stand, time] for flight in self.FLIGHTS]) <= 1
    
    def teletrap_can_be_used_on_stand(self, stand):
        return self.AIRCRAFT_STANDS_DATA['JetBridge_on_Arrival'][stand] != 'N' and self.AIRCRAFT_STANDS_DATA['JetBridge_on_Departure'][stand] != 'N'
    
    def two_wide_near_are_prohibited_func(self, model, stand, time):

        if stand - 1 in self.AIRCRAFT_STANDS:
            left_stand = sum([model.AS_occupied_time[flight, stand - 1, time] for flight in self.FLIGHTS_WIDE])
        else:
            left_stand = 0
        
        middle_stand = sum([model.AS_occupied_time[flight, stand, time] for flight in self.FLIGHTS_WIDE])

        if stand + 1 in self.AIRCRAFT_STANDS :
            right_stand = sum([model.AS_occupied_time[flight, stand + 1, time] for flight in self.FLIGHTS_WIDE])
        else:
            right_stand = 0
        
        return left_stand + middle_stand + right_stand <= 1
    
    def every_flight_must_have_its_stand_func(self, model, flight):
        return sum([self.model.AS_occupied[flight, stand] for stand in self.AIRCRAFT_STANDS]) == 1


    def make_model(self, start_dt=datetime(2019, 5, 17, 0, 0), end_dt=datetime(2019, 5, 17, 23, 55)):

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

        # занимаемые места (Рейс * МC) - переменные
        self.model.AS_occupied = pyo.Var(self.FLIGHTS, self.AIRCRAFT_STANDS, within=pyo.Binary, initialize=0)

        # занимаемые времена с учетом времени
        t = time.time()
        print(len(self.FLIGHTS), len(self.AIRCRAFT_STANDS), len(self.TIMES))
        self.model.AS_occupied_time = pyo.Expression(self.FLIGHTS, self.AIRCRAFT_STANDS, self.TIMES, rule=self.time_calculate_func)
        print('hehe I am here')
        elapsed = time.time() - t
        print(elapsed)
        # Cтоимость руления по аэродрому
        self.model.airport_taxiing_cost = pyo.Expression(self.FLIGHTS, rule=self.airport_taxiing_cost_func)

        # Стоимость использования МС ВС
        self.model.AS_using_cost = pyo.Expression(self.FLIGHTS, rule=self.AS_using_cost_def)

        # Стоимость использования перронных автобусов для посадки/высадки пассажиров
        self.model.busses_cost = pyo.Expression(self.FLIGHTS, rule=self.busses_cost_func)

        # Целевая переменная
        self.model.OBJ = pyo.Objective(expr=sum([self.model.airport_taxiing_cost[flight] for flight in self.FLIGHTS]) +\
                                            sum([self.model.AS_using_cost[stand] for stand in self.AIRCRAFT_STANDS]) +\
                                            sum([self.model.busses_cost[flight] for flight in self.FLIGHTS]), sense=pyo.minimize)
        # self.model.OBJ = pyo.Objective(expr=0, sense=pyo.minimize)

        self.model.only_one_flight_per_place = pyo.Constraint(self.AIRCRAFT_STANDS, self.TIMES, rule=self.only_one_flight_per_place_func)

        self.model.two_wide_near_are_prohibited = pyo.Constraint(self.AIRCRAFT_STANDS_WITH_TRAPS, self.TIMES, rule=self.two_wide_near_are_prohibited_func)

        self.model.every_flight_must_have_its_stand = pyo.Constraint(self.FLIGHTS, rule=self.every_flight_must_have_its_stand_func)


        self.opt_output = self.opt.solve(self.model, logfile='SOLVE_LOG', solnfile='SOLNFILE')
        print(self.opt_output)

    def get_model(self):
        return self.model

    def get_solution(self):
        assert self.model is not None
        AS_occupied_data = pd.DataFrame().from_dict(self.model.AS_occupied.extract_values(), orient='index', columns=[])
        best_stops['flight'] = best_stops.index.map(lambda x: x[0])
        best_stops['stand'] = best_stops.index.map(lambda x: x[1])
        self.chosen_frames = best_stops.sort_values(by='visual', ascending=True).loc[best_stops['best_stops'] == 1, 'stop'].to_list()
        return 
        

if __name__ == "__main__":
    d = DataExtended()
    
    opt = OptimizeDay(d)
    opt.make_model(datetime(2019, 5, 17, 0, 0), datetime(2019, 5, 17, 23, 55))
    o = opt.get_model()

    with open("opt.pkl", 'wb') as h:
        cloudpickle.dump(o, h)

    # with open("opt.pkl", 'rb') as h:
    #     tmp = cloudpickle.load(h)