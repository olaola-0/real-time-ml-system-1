from datetime import timedelta
from typing import Optional

import pandas as pd
from bokeh.plotting import figure


def plot_candles(
        df: pd.DataFrame,
        window_seconds: Optional[int] = 60,
        title: Optional[str] = '',
) -> 'Figure':
    """
    Genartes a Bokeh plot with candles from the given DataFrame.

    Args:
        df: DataFrame with the data to plot i.e. columns with open, high, low, close
        window_seconds: Size of the window in seconds
        title: Title of the plot

    Returns:
        figure.Figure: Bokeh figure with the candles plot and bolinger bands
    """
    # Convert the timestamp cloumn in unix seconds to a datetime object
    df['date'] = pd.to_datetime(df['timestamp'], unit='ms')

    inc = df.close > df.open
    dec = df.open > df.close
    w = 1000 * window_seconds / 2  # Band width in ms

    TOOLS = "pan,wheel_zoom,box_zoom,reset,save"

    x_max = df['date'].max() + timedelta(minutes=5)
    x_min = df['date'].min() - timedelta(minutes=5)
    p = figure(
        x_axis_type="datetime",
        tools=TOOLS,
        width=1000,
        title=title,
        x_range=(x_min, x_max),
    )
    p.grid.grid_line_alpha = 0.3

    p.segment(df.date, df.high, df.date, df.low, color="black")
    p.vbar(
        df.date[inc],
        w,
        df.open[inc],
        df.close[inc],
        fill_color="#70db40",
        line_color="black",
    )
    p.vbar(
        df.date[dec],
        w,
        df.open[dec],
        df.close[dec],
        fill_color="#F2583E",
        line_color="black",
    )

    return p
