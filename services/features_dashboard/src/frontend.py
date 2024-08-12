 # This is the frontend of the dashboard. It is a Streamlit app that will be served by the backend.
from time import sleep

import streamlit as st
from loguru import logger as logging

from src.backend import get_features_from_store
from src.plot import plot_candles


st.set_page_config(layout="wide")  # Use the wide layout
st.title("OHLC Features Dashboard")

online_or_offline = st.sidebar.selectbox(
    "Select the feature store to read from",
    ("online", "offline"),
)

# Add a date range slider to the sidebar
date_range = st.sidebar.date_input("Select date range")

with st.container():
    placeholder_chart = st.empty()

while True:
    # Load the data from the feature store
    data = get_features_from_store(online_or_offline)
    logging.debug(f'Recieved {len(data)} rows from the feature store')

    # Refresh the chart
    with placeholder_chart:
        st.bokeh_chart(plot_candles(data, title='OHLC Features'))

        sleep(15)