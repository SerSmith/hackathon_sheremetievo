import datetime
import streamlit as st
import pandas as pd
import plotly.express as px
import yaml
import plotly.graph_objects as go
from matplotlib.backends.backend_agg import RendererAgg


st.set_page_config(layout="wide")



with open("optimization_config.yaml", "rb") as h:
    config = yaml.safe_load(h)


select_scenario = st.sidebar.selectbox('Выбирите сценарий', config['dashboard_parameters']['solutions_paths'].keys())


def draw_dashboard(config, scenario_name):

    START_TIME = datetime.datetime(* config['task_config']['start_date'])
    END_TIME = datetime.datetime(* config['task_config']['end_date'])
    TIME_FORMAT = 'YYYY-MM-DD HH:mm:ss'

    st.title('Оптимизация расстановки самолетов по местам стоянок')
    st.header(scenario_name)


    data = pd.read_csv(config['dashboard_parameters']['solutions_paths'][scenario_name])
    data['Terminal'] = data['Terminal'].fillna('Away')

    slider_timesample = st.slider('', min_value=START_TIME, value=[START_TIME, END_TIME], max_value=END_TIME, format=TIME_FORMAT)

    timesample_cond = (
        (pd.to_datetime(data['flight_datetime']) >= slider_timesample[0])
        &
        (pd.to_datetime(data['flight_datetime']) <= slider_timesample[1])
        )

    data_timesample = data[timesample_cond]

    taxiing_cost = data_timesample['Taxiing_cost_solution'].sum()
    parking_cost = data_timesample['Parking_usage_cost_solution'].sum()
    bus_cost = data_timesample['Bus_usage_cost_solution'].sum()
    total_cost = taxiing_cost + parking_cost + bus_cost

    col1, col2, col3, col4 = st.columns(4)
    col1.metric('Стоимость руления', f'{taxiing_cost}')
    col2.metric('Стоимость МС', f'{parking_cost}')
    col3.metric('Стоимость автобусов', f'{bus_cost}')
    col4.metric('Общая стоимость', f'{total_cost}')

    col1, col2, col3 = st.columns([1,5,1])

    with col1:
        st.write("")

    with col2:
        fig = px.histogram(data_timesample,
                            x='Aircraft_Stand',
                            color='Terminal',
                            nbins=1000,
                            facet_row_spacing=0.5,
                            height=600,
                            width=800)
        fig.update_xaxes(title_text='Номер места стоянки')
        fig.update_yaxes(title_text='Рейсов за выбранный период')
        st.plotly_chart(fig)

    with col3:
        st.write("")



    _lock = RendererAgg.lock
    height = 250
    width = 500
    st.subheader('Statistics')
    st.markdown("##### **Выберите один из вариантов:**")
    variants = ['Количество рейсов', 'Количество клиентов']
    current_variant = []

    current_variant = st.selectbox('', variants)
    map_dict = {'Количество рейсов': {'flight_number': 'count'},'Количество клиентов': {'flight_PAX': 'sum'} }
    column = list(map_dict[current_variant].keys())[0]
    row1_space1, row1_1, row1_space2, row1_2, row1_space3 = st.columns((.1, 1, .1, 1, .1))


    jetbridge_df = data_timesample.groupby('JetBridge_can_be_used').agg(map_dict[current_variant]).reset_index()
    jetbridge_df['name'] = jetbridge_df['JetBridge_can_be_used'].map({False: 'Не используя телетрап',True: 'Используя телетрап'})
    jetbridge_df['procent'] = jetbridge_df[column]/sum(jetbridge_df[column]) * 100
    jetbridge_df['procent'] = jetbridge_df['procent'].apply(lambda x: str(round(x,1))+'%')

    with row1_1, _lock:
        st.subheader('Количество рейсов в зависимости от использования телетрапа')
        try:
            fig = go.Figure(go.Bar(
                x=list(jetbridge_df[column]),
                y=list(jetbridge_df['name']),
                text = list(jetbridge_df['procent']),
                textposition='auto',
                orientation='h',
                ))
            fig.update_layout(height=height, width=width, margin=dict(l=20, r=20, t=20, b=20))
            st.plotly_chart(fig)
        except:
            st.markdown("Cant't plot bar")

    jetbridge_df['JetBridge_on_Arrival'] = data_timesample['JetBridge_on_Arrival'].map({'D': 'Y', 'I': 'Y', 'N': 'N'})
    jetbridge_df = jetbridge_df.groupby('JetBridge_on_Arrival').agg(map_dict[current_variant]).reset_index()
    jetbridge_df['name'] = jetbridge_df['JetBridge_on_Arrival'].map({'Y': 'Контактное МС', 'N': 'Удаленное МС'})
    jetbridge_df['procent'] = jetbridge_df[column]/sum(jetbridge_df[column]) * 100
    jetbridge_df['procent'] = jetbridge_df['procent'].apply(lambda x: str(round(x,1))+'%')

    with row1_2, _lock:
        st.subheader('Количество рейсов через контактные / удаленные МС')
        try:
            fig = go.Figure(go.Bar(
                x=list(jetbridge_df[column]),
                y=list(jetbridge_df['name']),
                text = list(jetbridge_df['procent']),
                textposition='auto',
                orientation='h',
                ))
            fig.update_layout(height=height, width=width, margin=dict(l=20, r=20, t=20, b=20))
            st.plotly_chart(fig)
        except:
            st.markdown("Cant't plot bar")
    # flight_id_df = data_timesample.groupby('flight_ID').agg({'flight_number': 'count'}).reset_index()
    # flight_id_df['name'] = flight_id_df['flight_ID'].map({'D': 'Domestic', 'I': 'International'})
    # flight_id_df['procent'] = flight_id_df['flight_number']/sum(flight_id_df['flight_number']) * 100
    # flight_id_df['procent'] = flight_id_df['procent'].apply(lambda x: str(round(x,1))+'%')

    # with row1_2, _lock:
    #     st.subheader('Количество ВВЛ или МВЛ рейсов')
    #     try:
    #         fig = go.Figure(go.Bar(
    #             x=list(flight_id_df['flight_number']),
    #             y=list(flight_id_df['name']),
    #             text = list(flight_id_df['procent']),
    #             textposition='auto',
    #             orientation='h',
    #             ))
    #         fig.update_layout(height=height, width=width, margin=dict(l=20, r=20, t=20, b=20))
    #         st.plotly_chart(fig)
    #     except:
    #         st.markdown("Cant't plot bar")

    row2_space1, row2_1, row2_space2, row2_2, row2_space3 = st.columns(
        (.1, 1, .1, 1, .1))

    flight_id_df = data_timesample[data_timesample.JetBridge_can_be_used==True].groupby('flight_ID').agg({'flight_number': 'count'}).reset_index()
    flight_id_df['name'] = flight_id_df['flight_ID'].map({'D': 'Domestic', 'I': 'International'})
    flight_id_df['procent'] = flight_id_df['flight_number']/sum(flight_id_df['flight_number']) * 100
    flight_id_df['procent'] = flight_id_df['procent'].apply(lambda x: str(round(x,1))+'%')

    with row2_1, _lock:
        st.subheader('Количество МВЛ/ВВЛ рейсов c использованием телетрапа')
        try:
            fig = go.Figure(go.Bar(
                x=list(flight_id_df['flight_number']),
                y=list(flight_id_df['name']),
                text = list(flight_id_df['procent']),
                textposition='auto',
                orientation='h',
                ))
            fig.update_layout(height=height, width=width, margin=dict(l=20, r=20, t=20, b=20))
            st.plotly_chart(fig)
        except:
            st.markdown("Cant't plot bar")


    aircraft_class_df = data_timesample.groupby('Aircraft_Class').agg({'flight_number': 'count'}).reset_index()
    #aircraft_classs_df['name'] = aircraft_classs_df['Aircraft_Class'].map({'D': 'Domestic', 'I': 'International'})
    aircraft_class_df['procent'] = aircraft_class_df['flight_number']/sum(aircraft_class_df['flight_number']) * 100
    aircraft_class_df['procent'] = aircraft_class_df['procent'].apply(lambda x: str(round(x,1))+'%')

    with row2_2, _lock:
        st.subheader('Количество рейсов в зависимости от типа ВС \n ')
        try:
            fig = go.Figure(go.Bar(
                x=list(aircraft_class_df['flight_number']),
                y=list(aircraft_class_df['Aircraft_Class']),
                text = list(aircraft_class_df['procent']),
                textposition='auto',
                orientation='h',
                ))
            fig.update_layout(height=height, width=width, margin=dict(l=20, r=20, t=20, b=20))
            st.plotly_chart(fig)
        except:
            st.markdown("Cant't plot bar")

draw_dashboard(config, select_scenario)

