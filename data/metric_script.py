import sys
import os
import pandas as pd
import numpy as np
import math
from datetime import datetime, timedelta
TimeTable = pd.read_csv(sys.argv[1])
TGO = pd.read_csv(f"Handling_Time_Private.csv")
Lots = pd.read_csv(f"Aircraft_Stands_Private.csv")
TYPES = pd.read_csv(f"Aircraft_Classes_Private.csv")
Price = pd.read_csv(f"Handling_Rates_Private.csv")

TimeTable = TimeTable.rename(columns={'flight_terminal_#': 'flight_terminal'})
Lots = Lots.melt(
    id_vars=['Aircraft_Stand', 'JetBridge_on_Arrival', 'JetBridge_on_Departure', 'Terminal', 'Taxiing_Time'],
    value_vars=['1', '2', '3', '4', '5'], value_name="Bus_Time")
Lots = Lots.astype({'variable': 'int32'})
Lots = Lots.astype({'Bus_Time': 'int32'})
TimeTable = TimeTable.merge(Lots, how='inner', left_on=['Aircraft_Stand', 'flight_terminal'],
                            right_on=['Aircraft_Stand', 'variable'])

TYPES_ARRAY = []
num = 0
for i in range(0, 3):
    while num <= TYPES.Max_Seats[i]:
        TYPES_ARRAY.append([TYPES.Aircraft_Class[i], num])
        num += 1
TYPES_L = pd.DataFrame(TYPES_ARRAY, columns=['Aircraft_Class', 'Seats'])
TimeTable = TimeTable.merge(TYPES_L, how='inner', left_on=['flight_AC_PAX_capacity_total'], right_on=['Seats']).drop(
    ['Seats', 'variable'], axis=1)

JBs = []
for i in range(len(TimeTable)):
    if ((TimeTable.flight_AD[i] == "A" and TimeTable.flight_ID[i] == TimeTable.JetBridge_on_Arrival[i]) \
        or (TimeTable.flight_AD[i] == "D" and TimeTable.flight_ID[i] == TimeTable.JetBridge_on_Departure[i])) \
            and (TimeTable.Terminal[i] == TimeTable.flight_terminal[i]):
        JBs.append("JetBridge")
    else:
        JBs.append("Away")

TimeTable['JB'] = JBs
TimeTable = TimeTable.drop(['JetBridge_on_Arrival', 'JetBridge_on_Departure'], axis=1)

TGO = TGO.rename(columns={'JetBridge_Handling_Time': 'JetBridge', 'Away_Handling_Time': 'Away'})
TGO = TGO.melt(id_vars=['Aircraft_Class'], value_vars=['JetBridge', 'Away'], var_name="JB", value_name="Handling_Time")
# TGO.head()


TimeTable = TimeTable.merge(TGO, how='inner', left_on=['Aircraft_Class', 'JB'], right_on=['Aircraft_Class', 'JB'])
TimeTable = TimeTable.astype({'flight_datetime': 'datetime64[m]'})

Start_Times = []
End_Times = []
for i in range(len(TimeTable)):
    if TimeTable.flight_AD[i] == "A":
        Start_Times.append(TimeTable.flight_datetime[i] + np.timedelta64(TimeTable.Taxiing_Time[i], 'm'))
        End_Times.append(
            TimeTable.flight_datetime[i] + np.timedelta64(TimeTable.Taxiing_Time[i] + TimeTable.Handling_Time[i],
                                                          'm') - timedelta(minutes=1))
    else:
        Start_Times.append(
            TimeTable.flight_datetime[i] - np.timedelta64(TimeTable.Taxiing_Time[i] + TimeTable.Handling_Time[i], 'm'))
        End_Times.append(
            TimeTable.flight_datetime[i] - np.timedelta64(TimeTable.Taxiing_Time[i], 'm') - timedelta(minutes=1))


TimeTable['Start_Time'] = Start_Times
TimeTable['End_Time'] = End_Times
Price_ = pd.DataFrame(Price['Value'].values, index=Price.Name, columns=['Rate'])

Bus_Nums = []
Bus_Costs = []
Parking_Costs = []
Taxing_Costs = []
Total_Costs = []
for i in range(len(TimeTable)):
    Bus_Nums.append(math.ceil(TimeTable.flight_PAX[i] / 80))
    Bus_Costs.append(
        (TimeTable.JB[i] == 'Away') * TimeTable.Bus_Time[i] * Price_.loc['Bus_Cost_per_Minute'].Rate * Bus_Nums[i])
    Parking_Costs.append(((np.isnan(TimeTable.Terminal[i]) == True) * Price_.loc[
        'Away_Aircraft_Stand_Cost_per_Minute'].Rate + (np.isnan(TimeTable.Terminal[i]) != True) * Price_.loc[
                              'JetBridge_Aircraft_Stand_Cost_per_Minute'].Rate) *
                         TimeTable.Handling_Time[i])
    Taxing_Costs.append(TimeTable.Taxiing_Time[i] * Price_.loc['Aircraft_Taxiing_Cost_per_Minute'].Rate)
Total_Сosts = [x + y + z for x, y, z in zip(Bus_Costs, Parking_Costs, Taxing_Costs)]
TimeTable['Bus_Num'] = Bus_Nums
TimeTable['Bus_Costs'] = Bus_Costs
TimeTable['Parking_Costs'] = Parking_Costs
TimeTable['Taxing_Cost'] = Taxing_Costs
TimeTable['Total_Cost'] = Total_Сosts


print('Total cost: ', TimeTable.Total_Cost.sum())
score = float(TimeTable.Total_Cost.sum())

Check1 = pd.DataFrame(
    TimeTable[['Start_Time', 'End_Time', 'Aircraft_Stand', 'flight_AL_Synchron_code', 'flight_number']])
Check1['Minute'] = TimeTable.loc[:, 'Start_Time']
Checks1 = []
for i in range(len(Check1)):
    row = np.array(Check1.iloc[i].values)
    Checks1.append(np.append(row[0:5], row[5] + np.timedelta64(0, 'm')))
    while (row[5] < Check1.iloc[i][1]):
        Checks1.append(np.append(row[0:5], row[5] + np.timedelta64(1, 'm')))
        row[5] += np.timedelta64(1, 'm')


Check1 = pd.DataFrame(Checks1, columns=('Start', 'End', 'Aircraft_Stand', 'AL', 'Flight', 'Minute'))
Check1_ = Check1.groupby(['Minute', 'Aircraft_Stand']).count()
Check1_ = Check1_.rename(columns={"Start": "Count"}).drop(['End', 'AL', 'Flight'], axis=1)
Check_doubles = Check1_.loc[(Check1_['Count'] > 1)].count()[0]
assert Check_doubles == 0, 'Проверка 1 не пройдена. Обнаружено одновременное использование паркинга разными судами.'

intersect_df = Check1.groupby(['Minute', 'Aircraft_Stand'])['Flight'].apply(list).reset_index()
intersect_df['num'] = intersect_df.apply(lambda x: len(x.Flight), axis=1)
intersect_df = intersect_df[intersect_df['num'] > 1]
res_df = intersect_df.groupby(intersect_df['Flight'].map(tuple))['Aircraft_Stand'].count().reset_index().rename(
    columns={'Flight': 'Flights_intersection', 'Aircraft_Stand': 'Intersection_duration_min'})


Check2 = pd.DataFrame(TimeTable[['Start_Time', 'End_Time', 'Aircraft_Stand', 'flight_AL_Synchron_code', 'flight_number',
                                 'Aircraft_Class', 'Terminal']].loc[(TimeTable['Aircraft_Class'] == "Wide_Body")])

Check2['Minute'] = TimeTable.loc[:, 'Start_Time']
Checks2 = []
for i in range(len(Check2)):
    row = np.array(Check2.iloc[i].values)
    Checks2.append(np.append(row[0:7], row[7] + np.timedelta64(0, 'm')))
    while (row[7] < Check1.iloc[i][1]):
        Checks2.append(np.append(row[0:7], row[7] + np.timedelta64(1, 'm')))
        row[7] += np.timedelta64(1, 'm')


Check2 = \
    pd.DataFrame(Checks2, columns=('Start', 'End', 'Aircraft_Stand', 'AL', 'Flight', 'Class', 'Terminal', 'Minute'))[
        ['Minute', 'Terminal', 'Aircraft_Stand']]
Check2_ = pd.DataFrame(Check2.groupby(['Minute', 'Terminal'])['Aircraft_Stand'].apply(np.array).reset_index())

error = 0
for term in range(1, 6):
    temp = Check2_.loc[(Check2_['Terminal'] == term)]
    for i in range(len(temp)):
        for j in range(len(temp.iloc[i][2]) - 1):
            assert abs(temp.iloc[i][2][j] - temp.iloc[i][2][j + 1]) >= 2, 'Проверка 2 не пройдена. Обнаружен конфликт размера судов на соседних парковках.'


