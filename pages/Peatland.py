# -*- coding: utf-8 -*-
"""
Created on Jul 2022
Online Data Visualization of eddy covariance data using Dash
Peatland Senda Darwin - Chiloé
@author: David Trejo Cancino
"""

import dash
from dash import html, dcc, callback
from dash.dependencies import Input, Output
import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import numpy as np
import pathlib

dash.register_page(__name__)


# %% Data reading
# get relative data folder
PATH = pathlib.Path(__file__).parent
DATA_PATH = PATH.joinpath("../datasets").resolve()
df_fst = pd.read_csv(DATA_PATH.joinpath("peatland_daily_merged_flux.csv"), index_col='TIMESTAMP')  # Reading data
columns = df_fst.columns.to_list()  # Listing columns
df_fst = df_fst.astype(float)
columns = df_fst.columns.to_list()


#%% Data Processing Functions


def dew_point(T, RH):
    """
    Calculates Dew Point Temperature according to Magnus-Tetens formula
    described at Lawrence 2005. Uncertainties of 0.35°C for temperatures
    ranging from -45 to 60°C.

    Parameters
    ----------
    T : array of Series
        Ambient Temperature [°C].
    RH : array or Series
        Relative Humidity [%].

    Returns
    -------
    Td : array
        Dew Point Temperature.

    """
    a = 17.625
    b = 243.04  # °C
    alpha = np.log(RH / 100) + a * T / (b + T)
    Td = (b * alpha) / (a - alpha)
    return Td


def air_density(TA, PA):
    """air_density(TA, PA)
    Air density of moist air from air temperature and pressure.
    rho = PA / (Rd * TA)
    Parameters
    ----------
    TA : list or list like
        Air temperature (deg C)
    PA : list or list like
        Atmospheric pressure (kPa)
    Returns
    -------
    rho : list or list like
        air density (kg m-3)
    References
    ----------
    - Foken, T, 2008: Micrometeorology. Springer, Berlin, Germany.
    """
    kPa2Pa = 1000  # conversion kilopascal (kPa) to pascal (Pa)
    Rd = 287.0586  # gas constant of dry air (J kg-1 K-1) (Foken 2008 p. 245)
    Kelvin = 273.15  # conversion degree Celsius to Kelvin
    TA = TA + Kelvin
    PA = PA * kPa2Pa
    rho = PA / (Rd * TA)
    return (rho)


def concentration_to_mixing_ratio(rho, conc):
    """
    Converts H2O concentration to H2O mixing ratio

    Parameters
    ----------
    rho : Series
        Air density (kg m-3).
    conc : array
        H2O concentration (g m-3).

    Returns
    -------
    mix : Series
        H2O mixing ratio (g kg-1).

    """
    conc = pd.to_numeric(conc, errors='coerce')
    rho = pd.to_numeric(rho, errors='coerce')
    mix = conc / rho
    return mix


def latent_heat_vaporization(TA):
    """latent_heat_vaporization(TA)
    Latent heat of vaporization as a function of air temperature (deg C).
    Uses the formula: lmbd = (2.501 - 0.00237*Tair)10^6
    Parameters
    ----------
    TA : list or list like
        Air temperature (deg C)
    Returns
    -------
    lambda : list or list like
        Latent heat of vaporization (J kg-1)
    References
    ----------
    - Stull, B., 1988: An Introduction to Boundary Layer Meteorology (p.641)
      Kluwer Academic Publishers, Dordrecht, Netherlands
    - Foken, T, 2008: Micrometeorology. Springer, Berlin, Germany.
    """
    k1 = 2.501
    k2 = 0.00237
    lmbd = (k1 - k2 * TA) * 1e+06
    return (lmbd)


def LE_to_ET(LE, TA):
    """LE_to_ET(LE, TA)
    Convert LE (W m-2) to ET (kg m-2 s-1, aka mm s-1).
    Parameters
    ----------
    LE : list or list like
        Latent Energy (W m-2)
    TA : list or list like
        Air temperature (deg C)
    Returns
    -------
    ET : list or list like
        Evapotranspiration (kg m-2 s-1, aka mm s-1)"""
    lmbd = latent_heat_vaporization(TA)
    ET = LE / lmbd
    return (ET)


# %% Variables
# Time
date = df_fst.index
# date = df_fst['TIMESTAMP']
t = np.linspace(1, 24, 24)
# Diagnostics
uptime_anem = (df_fst['sonic_samples_Tot'] / 18000) * 100
uptime_irga = (df_fst['irga_samples_Tot'] / 18000) * 100
battery = df_fst['batt_volt_Avg']
# Wind
tau = df_fst['tau']
wind = df_fst['wnd_spd']
wind_dir = df_fst['wnd_dir_sonic']
u = df_fst['Ux_Avg']
v = df_fst['Uy_Avg']
w = df_fst['Uz_Avg']
ustar = df_fst['u_star']
# Energy
Hc = np.float64(df_fst['Hc'])
LE = np.float64(df_fst['LE'])
SW = np.float64(df_fst['NR01_NetRs_Avg'])
LW = np.float64(df_fst['NR01_NetRl_Avg'])
Rn = np.float64(SW) + np.float64(LW)
NETRAD = np.float64(df_fst['Rn_Avg'])
SHF_COMPONENTS = df_fst.filter(like='shf_Avg')
SHF = SHF_COMPONENTS.mean(axis=1)
CLOSURE = np.float64(Rn) - np.float64(SHF) - np.float64(LE) - np.float64(Hc)
# Fluxes and concentrations
FCO2 = df_fst['Fc'];
CO2 = df_fst['CO2_mean']
# fc_ux = df_fst['CO2_Ux_cov']; fc_uy = df_fst['CO2_Uy_cov']; fc_uz = df_fst['CO2_Uz_cov']
FH2O = df_fst['H2O_Uz_cov']
H2O = df_fst['H2O_mean']
CO2sig = df_fst['CO2_sig_strgth_mean']
H2Osig = df_fst['H2O_sig_strgth_mean']
# Met
Pa = df_fst['amb_press_mean']
Tc = df_fst['Tc_mean']
e = df_fst['e_tmpr_rh_mean']
esat = df_fst['e_sat_tmpr_rh_mean']
Ta = df_fst['T_tmpr_rh_mean']
pcell = df_fst['cell_press_mean']
pp = df_fst['Rain_mm_Tot'] * 48  # 48 samples per day
RH = df_fst['RH_tmpr_rh_mean']
Td = dew_point(Ta, RH)
r = df_fst['H2O_tmpr_rh_mean']
VPD = esat - e
rho = air_density(Ta, Pa)
r_irga = concentration_to_mixing_ratio(rho, H2O)
SWC_COMPONENTS = df_fst.filter(like='soil_water_T')
Tsoil_COMPONENTS = df_fst.filter(like='Tsoil_mean')
WTD_COMPONENTS = df_fst.filter(like='Level_')
WTD_max = WTD_COMPONENTS.filter(like='_Max')
WTD_min = WTD_COMPONENTS.filter(like='_Min')
ET = LE_to_ET(LE, Ta) * 60 * 60 * 24  # from seconds to daily vals
try:
    Globalrad = df_fst['GlobalRadiation']
    Diffrad = df_fst['DiffuseRadiation']
    Dirrad = Globalrad - Diffrad
    diff_bool = True
except KeyError:
    diff_bool = False


# %% function filtering the wind rose data based on the wind speed and direction (NEEDED BELOW)
def wind_dir_speed_freq(boundary_lower_speed, boundary_higher_speed, boundary_lower_direction,
                        boundary_higher_direction):
    # mask for wind speed column
    log_mask_speed = (wind_rose_data[:, 0] >= boundary_lower_speed) & (wind_rose_data[:, 0] < boundary_higher_speed)
    # mask for wind direction
    log_mask_direction = (wind_rose_data[:, 1] >= boundary_lower_direction) & (
                wind_rose_data[:, 1] < boundary_higher_direction)

    # application of the filter on the wind_rose_data array
    return wind_rose_data[log_mask_speed & log_mask_direction]


# %% Wind rose

# Creating a pandas dataframe with 8 wind speed bins for each of the 16 wind directions.
# dataframe structure: direction | strength | frequency (radius)

wind_rose_df_fst = pd.DataFrame(np.zeros((16 * 9, 3)), index=None, columns=('direction', 'strength', 'frequency'))

directions = ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE', 'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW']
directions_deg = np.array([0, 22.5, 45, 72.5, 90, 112.5, 135, 157.5, 180, 202.5, 225, 247.5, 270, 292.5, 315, 337.5])
speed_bins = ['0-2 m/s', '2-4 m/s', '4-6 m/s', '6-8 m/s', '8-10 m/s', '10-12 m/s', '12-14 m/s', '14-25 m/s', '>25 m/s']

# filling in the dataframe with directions and speed bins
wind_rose_df_fst.direction = directions * 9
wind_rose_df_fst.strength = np.repeat(speed_bins, 16)

# creating a multiindex dataframe with frequencies

idx = pd.MultiIndex.from_product([speed_bins,
                                  directions_deg],
                                 names=['wind_speed_bins', 'wind_direction_bins'])
col = ['frequency']
frequencies_df_fst = pd.DataFrame(0, idx, col)

wind_rose_data = np.asarray([np.float64(wind), np.float64(wind_dir)]).T
# meteo[['wind_speed_m/s', 'wind_direction_deg']].to_numpy()

# distance between the centre of the bin and its edge
step = 11.25

# converting data between 348.75 and 360 to negative
for i in range(len(wind_rose_data)):
    if directions_deg[-1] + step <= wind_rose_data[i, 1] and wind_rose_data[i, 1] < 360:
        wind_rose_data[i, 1] = wind_rose_data[i, 1] - 360

# determining the direction bins
bin_edges_dir = directions_deg - step
bin_edges_dir = np.append(bin_edges_dir, [directions_deg[-1] + step])

# determining speed bins ( the last bin is 50 as above those speeds the outliers were removed for the measurements)
threshold_outlier_rm = 50
bin_edges_speed = np.array([0, 2, 4, 6, 8, 10, 12, 14, 25, threshold_outlier_rm])

frequencies = np.array([])
# loop selecting given bins and calculating frequencies
for i in range(len(bin_edges_speed) - 1):
    for j in range(len(bin_edges_dir) - 1):
        bin_contents = wind_dir_speed_freq(bin_edges_speed[i], bin_edges_speed[i + 1], bin_edges_dir[j],
                                           bin_edges_dir[j + 1])

        # applying the filtering function for every bin and checking the number of measurements
        bin_size = len(bin_contents)
        frequency = bin_size / len(wind_rose_data)

        # obtaining the final frequencies of bin
        frequencies = np.append(frequencies, frequency)

# updating the frequencies dataframe
frequencies_df_fst.frequency = frequencies * 100  # [%]
wind_rose_df_fst.frequency = frequencies * 100  # [%]
# %% web app
fig_names = ['Battery', 'GHG Concentrations', 'Fluxes', 'Wind',
             'Wind Direction', 'Energy Balance', 'Biomet', 'Multi-channel signals',
             'Diffuse Radiation']

# layout = html.Div([
#     html.H1('Eddy Covariance Data - Forest Senda Darwin, Chiloé, IEB',
#             style={"textAlign": "center"}),

#     html.Div([
#         html.Div([
#             html.Pre(children="Variables", style={"fontSize":"150%"}),
#             dcc.Dropdown(
#                 id='fig-dropdown', value='Battery', clearable=False,
#                 persistence=True, persistence_type='session',
#                 options=[{'label': x, 'value': x} for x in fig_names])],
#             className='six columns'),], className='row'),

#     dcc.Graph(id='my-map', figure={}),
# ])
layout = html.Div([
    html.H1('Peatland Senda Darwin, Chiloé, IEB',
            style={"textAlign": "center"}),

    html.Div([
        html.Div([
            html.Pre(children="Variables", style={"fontSize": "150%"}),
            dcc.Dropdown(
                id='fig-dropdown2', value='Battery', clearable=False,
                persistence=True, persistence_type='session',
                options=[{'label': x, 'value': x} for x in fig_names])],
            className='six columns'), ], className='row'),

    dcc.Graph(id='my-map2', figure={}),
])


@callback(
    Output(component_id='my-map2', component_property='figure'),
    [Input(component_id='fig-dropdown2', component_property='value')])


def name_to_figure(fig_name):
    if fig_name == 'Battery':
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True)
        fig.add_trace(go.Scatter(x=date, y=battery, mode='lines', name='Battery Voltage'),
                      row=1, col=1)
        fig.update_yaxes(range=[10, 13], title_text="Voltage Input",
                         row=1, col=1)
        fig.add_trace(go.Scatter(x=date, y=uptime_anem, mode='lines', name='Uptime - anemometer'),
                      row=2, col=1)
        fig.add_trace(go.Scatter(x=date, y=uptime_irga, mode='lines', name='Uptime - IRGA'),
                      row=2, col=1)
        fig.update_yaxes(range=[0, 103], title_text="Uptime in 30 minutes [%]",
                         row=2, col=1)
        fig.update_layout(autosize=False, width=1750, height=800)
        # Slider
        fig.update_layout(
            xaxis=dict(
                rangeselector=dict(
                    buttons=list([
                        dict(count=1,
                             label="1m",
                             step="month",
                             stepmode="backward"),
                        dict(count=6,
                             label="6m",
                             step="month",
                             stepmode="backward"),
                        dict(count=1,
                             label="YTD",
                             step="year",
                             stepmode="todate"),
                        dict(count=1,
                             label="1y",
                             step="year",
                             stepmode="backward"),
                        dict(step="all")
                    ])),
            ),
            xaxis_showticklabels=True, xaxis2_showticklabels=True,
            xaxis2_rangeslider_visible=True, xaxis2_type="date",
            xaxis2_rangeslider_bgcolor='grey', xaxis2_rangeslider_thickness=0.03)
    elif fig_name == 'GHG Concentrations':
        fig = make_subplots(rows=2, cols=1, specs=[[{"secondary_y": True}],
                                                   [{"secondary_y": True}]], shared_xaxes=True)
        # CO2
        fig.add_trace(go.Scatter(x=date, y=CO2, mode='lines', name='CO2',
                                 line=dict(color='blue')), row=1, col=1)
        fig.update_yaxes(range=[250, 1200], title_text="CO2 [mg m-3]",
                         row=1, col=1, titlefont=dict(color="blue"),
                         tickfont=dict(color="blue"))
        fig.add_trace(go.Scatter(x=date, y=CO2sig, mode='lines', name='CO2 signal strength',
                                 line=dict(color='red')), secondary_y=True, row=1, col=1)
        fig.update_yaxes(range=[0.1, 1.1], title_text="CO2 signal strength",
                         row=1, col=1, secondary_y=True, titlefont=dict(color="red"),
                         tickfont=dict(color="red"))
        # H2O
        fig.add_trace(go.Scatter(x=date, y=H2O, mode='lines', name='H2O',
                                 line=dict(color='green')),
                      row=2, col=1)
        fig.update_yaxes(range=[1, 20], title_text="H2O [g m-3]",
                         row=2, col=1, titlefont=dict(color="green"),
                         tickfont=dict(color="green"))
        fig.add_trace(go.Scatter(x=date, y=H2Osig, mode='lines', name='H2O signal strength',
                                 line=dict(color='purple')),
                      row=2, col=1, secondary_y=True)
        fig.update_yaxes(range=[0.1, 1.1], title_text="H2O signal strength",
                         row=2, col=1, secondary_y=True, titlefont=dict(color="purple"),
                         tickfont=dict(color="purple"))
        # fig.update_layout(autosize=False, width=1750, height=500)
        fig.update_layout(autosize=False, width=1750, height=800)
        # Slider
        fig.update_layout(
            xaxis=dict(
                rangeselector=dict(
                    buttons=list([
                        dict(count=1,
                             label="1m",
                             step="month",
                             stepmode="backward"),
                        dict(count=6,
                             label="6m",
                             step="month",
                             stepmode="backward"),
                        dict(count=1,
                             label="YTD",
                             step="year",
                             stepmode="todate"),
                        dict(count=1,
                             label="1y",
                             step="year",
                             stepmode="backward"),
                        dict(step="all")
                    ])),
            ),
            xaxis_showticklabels=True, xaxis2_showticklabels=True,
            xaxis2_rangeslider_visible=True, xaxis2_type="date",
            xaxis2_rangeslider_bgcolor='grey', xaxis2_rangeslider_thickness=0.03)
    elif fig_name == 'Fluxes':
        fig = make_subplots(rows=4, cols=1, specs=[[{"secondary_y": True}],
                                                   [{"secondary_y": True}], [{"secondary_y": True}],
                                                   [{"secondary_y": True}]], shared_xaxes=True)
        # CO2
        fig.add_trace(go.Scatter(x=date, y=FCO2, mode='lines', name='CO2 Flux',
                                 line=dict(color='blue')), row=1, col=1)
        fig.update_yaxes(range=[-5, 5], title_text="CO2 Flux [mg s-1 m-2]",
                         row=1, col=1, titlefont=dict(color="blue"),
                         tickfont=dict(color="blue"))
        fig.add_trace(go.Scatter(x=date, y=CO2sig, mode='lines', name='CO2 signal strength',
                                 line=dict(color='red')), secondary_y=True, row=1, col=1)
        fig.update_yaxes(range=[0.3, 1.1], title_text="CO2 signal strength",
                         row=1, col=1, secondary_y=True, titlefont=dict(color="red"),
                         tickfont=dict(color="red"))
        # H2O
        fig.add_trace(go.Scatter(x=date, y=FH2O, mode='lines', name='H2O Flux',
                                 line=dict(color='green')),
                      row=2, col=1)
        fig.update_yaxes(range=[-0.4, 0.4], title_text="H2O Flux [g s-1 m-2]",
                         row=2, col=1, titlefont=dict(color="green"),
                         tickfont=dict(color="green"))
        fig.add_trace(go.Scatter(x=date, y=H2Osig, mode='lines', name='H2O signal strength',
                                 line=dict(color='purple')),
                      row=2, col=1, secondary_y=True)
        fig.update_yaxes(range=[0.3, 1.1], title_text="H2O signal strength",
                         row=2, col=1, secondary_y=True, titlefont=dict(color="purple"),
                         tickfont=dict(color="purple"))
        # ET
        fig.add_trace(go.Scatter(x=date, y=ET, mode='lines', name='Evapotranspiration (ET)'),
                      row=3, col=1)
        fig.update_yaxes(range=[0, 7], title_text="ET [mm day-1]",
                         row=3, col=1)
        # Momentum flux
        fig.add_trace(go.Scatter(x=date, y=tau, mode='lines', name='Momentum Flux (Tau)'),
                      row=4, col=1)
        fig.update_yaxes(range=[0, 3], title_text="Tau [kg s-2 m-1]",
                         row=4, col=1)
        # Slider
        fig.update_layout(
            xaxis=dict(
                rangeselector=dict(
                    buttons=list([
                        dict(count=1,
                             label="1m",
                             step="month",
                             stepmode="backward"),
                        dict(count=6,
                             label="6m",
                             step="month",
                             stepmode="backward"),
                        dict(count=1,
                             label="YTD",
                             step="year",
                             stepmode="todate"),
                        dict(count=1,
                             label="1y",
                             step="year",
                             stepmode="backward"),
                        dict(step="all")
                    ])),
            ),
            xaxis_showticklabels=True, xaxis2_showticklabels=True,
            xaxis3_showticklabels=True, xaxis4_showticklabels=True,
            xaxis4_rangeslider_visible=True, xaxis4_type="date",
            xaxis4_rangeslider_bgcolor='grey', xaxis4_rangeslider_thickness=0.03)
        fig.update_layout(autosize=False, width=1750, height=1100)
    elif fig_name == 'Wind':
        fig = make_subplots(rows=4, cols=1, specs=[[{"secondary_y": True}],
                                                   [{"secondary_y": True}],
                                                   [{"secondary_y": True}],
                                                   [{"secondary_y": True}]])
        # Wind Components
        fig.add_trace(go.Scatter(x=date, y=u, mode='lines', name='Ux',
                                 line=dict(color='royalblue')), row=1, col=1)
        fig.add_trace(go.Scatter(x=date, y=v, mode='lines', name='Uy',
                                 line=dict(color='tomato')), row=1, col=1)
        fig.add_trace(go.Scatter(x=date, y=w, mode='lines', name='Uz',
                                 line=dict(color='saddlebrown')), row=1, col=1,
                      secondary_y=True)
        fig.update_yaxes(range=[-9, 9], title_text="Ux, Uy [m/s]",
                         row=1, col=1, titlefont=dict(color="royalblue"),
                         tickfont=dict(color="royalblue"))
        fig.update_yaxes(range=[-2, 2], title_text="Uz [m/s]",
                         row=1, col=1, titlefont=dict(color="saddlebrown"),
                         tickfont=dict(color="saddlebrown"), secondary_y=True)
        fig.update_xaxes(row=1, col=1, matches='x')
        # Wind Speed and friction velocity
        fig.add_trace(go.Scatter(x=date, y=wind, mode='lines',
                                 name='Wind Speed', line=dict(color='grey')),
                      row=2, col=1)
        fig.add_trace(go.Scatter(x=date, y=ustar, mode='lines',
                                 name='Friction Velocity', line=dict(color='orange')),
                      row=2, col=1, secondary_y=True)
        fig.update_layout(autosize=False, width=1750, height=1500)
        fig.update_yaxes(range=[0, 10], title_text="Wind Speed [m/s]",
                         row=2, col=1, titlefont=dict(color="grey"),
                         tickfont=dict(color="teal"))
        fig.update_yaxes(range=[0, 2], title_text="Friction Velocity [m/s]",
                         row=2, col=1, titlefont=dict(color="orange"),
                         tickfont=dict(color="orange"), secondary_y=True)
        fig.update_xaxes(row=2, col=1, matches='x')
        # Wind Direction Series
        fig.add_trace(go.Bar(x=date, y=wind_dir,
                             name='Wind Direction Time Series', marker_color='forestgreen'),
                      row=3, col=1)
        fig.update_yaxes(range=[-182, 182],
                         title_text="Degrees from North", row=3, col=1)
        fig.update_xaxes(row=3, col=1, matches='x')
        # Wind direction Histogram
        fig.add_trace(go.Histogram(x=wind_dir, histnorm='probability',
                                   marker_color='#330C73', name='Wind Direction Histogram'),
                      row=4, col=1)
        fig.update_yaxes(title_text="Normalized Probability", row=4, col=1)
        fig.update_xaxes(title_text="Degrees from North", row=4, col=1)
        # Slider
        fig.update_layout(
            xaxis=dict(
                rangeselector=dict(
                    buttons=list([
                        dict(count=1,
                             label="1m",
                             step="month",
                             stepmode="backward"),
                        dict(count=6,
                             label="6m",
                             step="month",
                             stepmode="backward"),
                        dict(count=1,
                             label="YTD",
                             step="year",
                             stepmode="todate"),
                        dict(count=1,
                             label="1y",
                             step="year",
                             stepmode="backward"),
                        dict(step="all")
                    ])),
            ),
            xaxis3_rangeslider_visible=True, xaxis3_type="date",
            xaxis3_rangeslider_bgcolor='grey', xaxis3_rangeslider_thickness=0.03)
        fig.update_layout(autosize=False, width=1750, height=1200)
    elif fig_name == 'Wind Direction':
        fig = go.Figure()
        fig.add_trace(go.Barpolar(
            r=frequencies_df_fst.loc[('0-2 m/s'), 'frequency'],
            name='0-2 m/s',
            marker_color='#482878'))

        fig.add_trace(go.Barpolar(
            r=frequencies_df_fst.loc[('2-4 m/s'), 'frequency'],
            name='2-4 m/s',
            marker_color='#3e4989'))

        fig.add_trace(go.Barpolar(
            r=frequencies_df_fst.loc[('4-6 m/s'), 'frequency'],
            name='4-6 m/s',
            marker_color='#31688e'))

        fig.add_trace(go.Barpolar(
            r=frequencies_df_fst.loc[('6-8 m/s'), 'frequency'],
            name='6-8 m/s',
            marker_color='#26828e'))

        fig.add_trace(go.Barpolar(
            r=frequencies_df_fst.loc[('8-10 m/s'), 'frequency'],
            name='8-10 m/s',
            marker_color='#1f9e89'))

        fig.add_trace(go.Barpolar(
            r=frequencies_df_fst.loc[('10-12 m/s'), 'frequency'],
            name='10-12 m/s',
            marker_color='#35b779'))

        fig.add_trace(go.Barpolar(
            r=frequencies_df_fst.loc[('12-14 m/s'), 'frequency'],
            name='12-14 m/s',
            marker_color='#6ece58'))

        fig.add_trace(go.Barpolar(
            r=frequencies_df_fst.loc[('14-25 m/s'), 'frequency'],
            name='14-25 m/s',
            marker_color='#b5de2b'))

        fig.add_trace(go.Barpolar(
            r=frequencies_df_fst.loc[('>25 m/s'), 'frequency'],
            name='>25 m/s',
            marker_color='#fde725'))

        fig.update_traces(text=['North', 'NNE', 'NE', 'ENE', 'East', 'ESE',
                                'SE', 'SSE', 'South', 'SSW', 'SW', 'WSW', 'West', 'WNW', 'NW', 'NNW'])

        fig.update_layout(
            autosize=False, width=1500, height=800,
            title='Wind Rose',
            title_font_size=26,
            title_x=0.463,
            legend_font_size=18,
            polar_radialaxis_ticksuffix='%',
            polar_angularaxis_rotation=90,
            polar_angularaxis_direction='clockwise',
            polar_angularaxis_tickmode='array',
            polar_angularaxis_tickvals=[0, 22.5, 45, 72.5, 90, 112.5, 135,
                                        157.5, 180, 202.5, 225, 247.5, 270,
                                        292.5, 315, 337.5],
            polar_angularaxis_ticktext=['<b>North</b>', 'NNE', '<b>NE</b>',
                                        'ENE', '<b>East</b>', 'ESE', '<b>SE</b>',
                                        'SSE', '<b>South</b>', 'SSW', '<b>SW</b>',
                                        'WSW', '<b>West</b>', 'WNW', '<b>NW</b>',
                                        'NNW'],
            polar_angularaxis_tickfont_size=22,
            polar_radialaxis_tickmode='linear',
            polar_radialaxis_angle=45,
            polar_radialaxis_tick0=5,
            polar_radialaxis_dtick=5,
            polar_radialaxis_tickangle=100,
            polar_radialaxis_tickfont_size=14)

    elif fig_name == 'Energy Balance':
        # 30 min data of energy components
        fig = make_subplots(rows=2, cols=1, specs=[[{"secondary_y": True}],
                                                   [{"secondary_y": True}]], shared_xaxes=True)
        fig.add_trace(go.Scatter(x=date, y=SW, mode='lines',
                                 name='Net Short Wave Radiation', line=dict(color='blue')), row=1, col=1)
        fig.add_trace(go.Scatter(x=date, y=LW, mode='lines',
                                 name='Net Long Wave Radiation', line=dict(color='orange')), row=1, col=1)
        fig.add_trace(go.Scatter(x=date, y=NETRAD, mode='lines',
                                 name='NET Radiation from NR Lite', line=dict(color='red')), row=1, col=1)
        fig.add_trace(go.Scatter(x=date, y=Rn, mode='lines',
                                 name='Net Radiation from NR01 (4 components)', line=dict(color='green')),
                      row=1, col=1)
        fig.add_trace(go.Scatter(x=date, y=LE, mode='lines',
                                 name='Latent Heat', line=dict(color='purple')), row=1, col=1)
        fig.add_trace(go.Scatter(x=date, y=Hc, mode='lines',
                                 name='Sensible Heat', line=dict(color='brown')), row=1, col=1)
        fig.add_trace(go.Scatter(x=date, y=SHF, mode='lines',
                                 name='Soil Heat Flux', line=dict(color='pink')), row=1, col=1)
        fig.update_yaxes(range=[-500, 1300], title_text="Energy Components [W/m2]",
                         row=1, col=1)
        #  30 min data of energy balance Closure
        fig.add_trace(go.Scatter(x=date, y=CLOSURE, mode='lines', name='CLOSURE',
                                 line=dict(color='gray')), row=2, col=1)
        fig.update_yaxes(range=[-700, 700], title_text="Closure NetRad - SHF - LE - H [W/m2]",
                         row=2, col=1)
        # Slider
        fig.update_layout(
            xaxis=dict(
                rangeselector=dict(
                    buttons=list([
                        dict(count=1,
                             label="1m",
                             step="month",
                             stepmode="backward"),
                        dict(count=6,
                             label="6m",
                             step="month",
                             stepmode="backward"),
                        dict(count=1,
                             label="YTD",
                             step="year",
                             stepmode="todate"),
                        dict(count=1,
                             label="1y",
                             step="year",
                             stepmode="backward"),
                        dict(step="all")
                    ])),
            ),
            xaxis_showticklabels=True, xaxis2_showticklabels=True,
            xaxis2_rangeslider_visible=True, xaxis2_type="date",
            xaxis2_rangeslider_bgcolor='grey', xaxis2_rangeslider_thickness=0.03)
        fig.update_layout(autosize=False, width=1750, height=1000)
    elif fig_name == 'Biomet':
        # 30 min data of energy components
        fig = make_subplots(rows=4, cols=1, specs=[[{"secondary_y": True}],
                                                   [{"secondary_y": True}], [{"secondary_y": True}],
                                                   [{"secondary_y": True}]],
                            shared_xaxes=True)
        # Temperature
        fig.add_trace(go.Scatter(x=date, y=Ta, mode='lines', name='Ambient Temperature',
                                 line=dict(color='blue')), row=1, col=1)
        fig.add_trace(go.Scatter(x=date, y=Td, mode='lines', name='Dew Point',
                                 line=dict(color='orange')), row=1, col=1)
        fig.update_yaxes(range=[-10, 32], title_text="°C",
                         row=1, col=1)
        # Humidity
        fig.add_trace(go.Scatter(x=date, y=RH, mode='lines', name='Relative Humidity HMP',
                                 line=dict(color='blue')), row=2, col=1)
        fig.add_trace(go.Scatter(x=date, y=r, mode='lines', name='H2O Mixing Ratio HMP',
                                 line=dict(color='orange')), row=2, col=1, secondary_y=True)
        fig.add_trace(go.Scatter(x=date, y=r_irga, mode='lines', name='H2O Mixing Ratio IRGA',
                                 line=dict(color='peru')), row=2, col=1, secondary_y=True)
        fig.update_yaxes(range=[15, 105], title_text="[%]",
                         row=2, col=1, titlefont=dict(color="blue"),
                         tickfont=dict(color="blue"))
        fig.update_yaxes(range=[0, 20], title_text="[g/kg]",
                         row=2, col=1, titlefont=dict(color="orange"),
                         tickfont=dict(color="orange"), secondary_y=True)
        # Pressure
        fig.add_trace(go.Scatter(x=date, y=Pa, mode='lines', name='Ambient Pressure'),
                      row=3, col=1)

        fig.add_trace(go.Scatter(x=date, y=pcell, mode='lines', name='Cell Pressure'),
                      row=3, col=1)
        fig.update_yaxes(range=[95, 110], title_text="[kPa]",
                         row=3, col=1)
        # Precipitation
        fig.add_trace(go.Scatter(x=date, y=pp, mode='lines', name='Precipitation'),
                      row=4, col=1)
        fig.update_yaxes(range=[0, 100], title_text="[mm]",
                         row=4, col=1)
        # Slider
        fig.update_layout(
            xaxis=dict(
                rangeselector=dict(
                    buttons=list([
                        dict(count=1,
                             label="1m",
                             step="month",
                             stepmode="backward"),
                        dict(count=6,
                             label="6m",
                             step="month",
                             stepmode="backward"),
                        dict(count=1,
                             label="YTD",
                             step="year",
                             stepmode="todate"),
                        dict(count=1,
                             label="1y",
                             step="year",
                             stepmode="backward"),
                        dict(step="all")
                    ])),
            ),
            xaxis_showticklabels=True, xaxis2_showticklabels=True, xaxis3_showticklabels=True,
            xaxis4_showticklabels=True,
            xaxis4_rangeslider_visible=True, xaxis4_type="date",
            xaxis4_rangeslider_bgcolor='grey', xaxis4_rangeslider_thickness=0.03)
        fig.update_layout(autosize=False, width=1750, height=1500)
    elif fig_name == 'Multi-channel signals':
        fig = make_subplots(rows=4, cols=1)
        for i in range(np.shape(SHF_COMPONENTS)[1]):
            shf = SHF_COMPONENTS.iloc[:, i]
            fig.add_trace(go.Scatter(x=date, y=shf, mode='lines',
                                     name='SHF ' + str(i + 1)), row=1, col=1)
            fig.update_yaxes(range=[-30, 30], title_text="Soil Heat Flux [W m-2]",
                             row=1, col=1)
        for i in range(np.shape(SWC_COMPONENTS)[1]):
            swc = SWC_COMPONENTS.iloc[:, i]
            fig.add_trace(go.Scatter(x=date, y=swc, mode='lines',
                                     name='SWC ' + str(i + 1)), row=2, col=1)
            fig.update_yaxes(range=[0, 1.4], title_text="Soil Water Content [m3 m-3]",
                             row=2, col=1)
        for i in range(np.shape(Tsoil_COMPONENTS)[1]):
            Tsoil = Tsoil_COMPONENTS.iloc[:, i]
            fig.add_trace(go.Scatter(x=date, y=Tsoil, mode='lines',
                                     name='Tsoil ' + str(i + 1)), row=3, col=1)
            fig.update_yaxes(range=[-2, 28], title_text="Soil Temperature [°C]",
                             row=3, col=1)
        for i in range(np.shape(WTD_max)[1]):
            wtd_max = WTD_max.iloc[:, i]
            fig.add_trace(go.Scatter(x=date, y=wtd_max, mode='lines',
                                     name='WTD Max ' + str(i + 1)), row=4, col=1)
            fig.update_yaxes(range=[-0.1, 2], title_text="Water Table Depth[m]",
                             row=4, col=1)
        for i in range(np.shape(WTD_min)[1]):
            wtd_min = WTD_min.iloc[:, i]
            fig.add_trace(go.Scatter(x=date, y=wtd_min, mode='lines',
                                     name='WTD Min ' + str(i + 1)), row=4, col=1)

        # Slider
        fig.update_layout(
            xaxis=dict(
                rangeselector=dict(
                    buttons=list([
                        dict(count=1,
                             label="1m",
                             step="month",
                             stepmode="backward"),
                        dict(count=6,
                             label="6m",
                             step="month",
                             stepmode="backward"),
                        dict(count=1,
                             label="YTD",
                             step="year",
                             stepmode="todate"),
                        dict(count=1,
                             label="1y",
                             step="year",
                             stepmode="backward"),
                        dict(step="all")
                    ])),
            ),
            xaxis_showticklabels=True, xaxis2_showticklabels=True, xaxis3_showticklabels=True,
            xaxis4_showticklabels=True,
            xaxis4_rangeslider_visible=True, xaxis4_type="date",
            xaxis4_rangeslider_bgcolor='grey', xaxis4_rangeslider_thickness=0.03)
        fig.update_layout(autosize=False, width=1750, height=1500)
    elif fig_name == 'Diffuse Radiation':
        if diff_bool:
            fig = make_subplots(rows=1, cols=1)
            # Diffuse, global and direct radiation
            fig.add_trace(go.Scatter(x=date, y=Globalrad, mode='lines',
                                     name='Global Radiation'), row=1, col=1)
            fig.add_trace(go.Scatter(x=date, y=Diffrad, mode='lines',
                                     name='Diffuse Radiation'), row=1, col=1)
            fig.add_trace(go.Scatter(x=date, y=Dirrad, mode='lines',
                                     name='Direct Radiation'), row=1, col=1)
            fig.update_yaxes(range=[-10, 1000], title_text="°C",
                             row=1, col=1)
            # Slider
            fig.update_layout(
                xaxis=dict(
                    rangeselector=dict(
                        buttons=list([
                            dict(count=1,
                                 label="1m",
                                 step="month",
                                 stepmode="backward"),
                            dict(count=6,
                                 label="6m",
                                 step="month",
                                 stepmode="backward"),
                            dict(count=1,
                                 label="YTD",
                                 step="year",
                                 stepmode="todate"),
                            dict(count=1,
                                 label="1y",
                                 step="year",
                                 stepmode="backward"),
                            dict(step="all")
                        ])),
                ),
                xaxis_showticklabels=True, xaxis_rangeslider_visible=True,
                xaxis_type="date", xaxis_rangeslider_bgcolor='grey',
                xaxis_rangeslider_thickness=0.03)
            fig.update_layout(width=1750, height=700)
        else:
            fig = go.Figure()
            fig.add_annotation(dict(font=dict(color='red', size=20),
                                    x=0,
                                    y=-0.12,
                                    showarrow=False,
                                    text="No diffuse radiation instrument on this station",
                                    textangle=0,
                                    xanchor='left',
                                    xref="paper",
                                    yref="paper"))
            fig.update_layout(width=800, height=700)
    return fig 