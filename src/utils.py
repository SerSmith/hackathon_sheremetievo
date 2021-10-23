from datetime import timedelta

def teletrap_can_be_used(flight, stand, FLIGHTS_DATA, AIRCRAFT_STANDS_DATA):
    """Можно ли использовать телетрап на этом рейсе и на этом месте

    Args:
        flight (str): номер рейса
        stand (str): МС
        FLIGHTS_DATA (dict): Данные рейсов
        AIRCRAFT_STANDS_DATA(dict): Данные МС
    """
    
    # Телетрап на данном МС доступен только в случае, если:
    # 1)терминал рейса соответствует терминалу МC
    # 2)значение поля flight_ID рейса (метка МВЛ/ВВЛ – Domestic/Intern£tion£l) совпадает с соответствующей меткой поля JetBridge_on_Arriv£l (для прилетающих рейсов) или JetBridge_on_Dep£rture (для вылетающих рейсов) МС

    cond1 = FLIGHTS_DATA['flight_terminal_#'][flight] == AIRCRAFT_STANDS_DATA['Terminal'][stand]
    cond2 = ((FLIGHTS_DATA['flight_ID'][flight] == AIRCRAFT_STANDS_DATA['JetBridge_on_Arrival'][stand])\
              and (FLIGHTS_DATA['flight_AD'][flight] == 'A'))\
                or\
            ((FLIGHTS_DATA['flight_ID'][flight] == AIRCRAFT_STANDS_DATA['JetBridge_on_Departure'][stand]) and\
            (FLIGHTS_DATA['flight_AD'][flight] == 'D'))

    return cond1 and cond2

def time_calculate_func(flight,
                        aircraft_stand,
                        time,
                        FLIGHTS_DATA,
                        AIRCRAFT_STANDS_DATA,
                        HANGLING_TIME,
                        time_box):
    """Расчет занято ли место рейсом в это время

    Args:
        flight (str): рейс
        aircraft_stand (str): МС
        time (str): временной интервал
        FLIGHTS_DATA (dict): данные о рейсах
        AIRCRAFT_STANDS_DATA (dict): данные о МС
        HANGLING_TIME (dict): данные о армени процедур
        time_box: int - кол-во минут
    """
    flight_time = FLIGHTS_DATA['flight_datetime'][flight]
    taxiing_time = int(AIRCRAFT_STANDS_DATA['Taxiing_Time'][aircraft_stand])
    arrival_or_depature = FLIGHTS_DATA['flight_AD'][flight]
    use_trap_flg = teletrap_can_be_used(flight, aircraft_stand, FLIGHTS_DATA, AIRCRAFT_STANDS_DATA)

    if use_trap_flg:
        column_handling_time = 'JetBridge_Handling_Time'
    else: 
        column_handling_time = 'Away_Handling_Time'
    aircraft_class = FLIGHTS_DATA['aircraft_class'][flight]
    handling_time = HANGLING_TIME[column_handling_time][aircraft_class]

    if arrival_or_depature == 'D':
        time_start = flight_time - timedelta(minutes=handling_time) - timedelta(minutes=taxiing_time)
        time_end = flight_time - timedelta(minutes=taxiing_time)
    elif arrival_or_depature == 'A':
        time_start = flight_time + timedelta(minutes=taxiing_time)
        time_end = flight_time + timedelta(minutes=handling_time) + timedelta(minutes=taxiing_time)
    else:
        raise ValueError(f"arrival_or_depature имеет некорректное значение: {arrival_or_depature} , а должно быть A или D")
    
    if max(time_start, time) > min(time_end, time + timedelta(minutes=time_box)):
        result = 0
    else:
        result = 1
    return result

def teletrap_can_be_used_on_stand(stand, AIRCRAFT_STANDS_DATA):
    """Можно ли использовать телетрап на этом мс

    Args:
        stand (str): МС
        AIRCRAFT_STANDS_DATA (dict): Данные по МС
    """
    return AIRCRAFT_STANDS_DATA['JetBridge_on_Arrival'][stand] != 'N' or AIRCRAFT_STANDS_DATA['JetBridge_on_Departure'][stand] != 'N'