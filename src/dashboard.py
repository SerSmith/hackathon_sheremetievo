mport datetime
import streamlit as st
import pandas as pd
import plotly.express as px

df = px.data.tips()

DATA_PATH = '/Users/affernus/Notebooks_and_py/SVO/result.csv'
START_TIME = datetime.datetime(2019, 5, 17, 00, 00, 00)
END_TIME = datetime.datetime(2019, 5, 17, 23, 59, 59)
TIME_FORMAT = 'YYYY-MM-DD HH:mm:ss'

st.title('Оптимизация расстановки самолетов по местам стоянок')

data = pd.read_csv(DATA_PATH)

data['Terminal'] = data['Terminal'].fillna('Away')

slider_timesample = st.slider('',min_value=START_TIME, value=[START_TIME, END_TIME], max_value=END_TIME, format=TIME_FORMAT)

timesample_cond = (
    (pd.to_datetime(data['start_parking']) >= slider_timesample[0])
    &
    (pd.to_datetime(data['start_parking']) <= slider_timesample[1])
    )

data_timesample = data[timesample_cond]

taxiing_cost = data_timesample['Taxiing_cost'].sum()
parking_cost = data_timesample['Parking_usage_cost'].sum()
bus_cost = data_timesample['Bus_usage_cost'].sum()
total_cost = taxiing_cost + parking_cost + bus_cost

col1, col2, col3, col4 = st.columns(4)
col1.metric('Стоимость руления', f'{taxiing_cost}')
col2.metric('Стоимость МС', f'{parking_cost}')
col3.metric('Стоимость автобусов', f'{bus_cost}')
col4.metric('Общая стоимость', f'{total_cost}')

fig = px.histogram(data_timesample, x='Aircraft_Stand', color='Terminal', nbins=1000, facet_row_spacing=0.5)
fig.update_xaxes(title_text='Номер места стоянки')
fig.update_yaxes(title_text='Рейсов за выбранный период')

st.plotly_chart(fig)
