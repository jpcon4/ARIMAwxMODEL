"""
PART 1: WEB SCRAPING

This part of the code is able to take a url of a timeseries viewer through the
NWS Website and is able to spit out the past 720 hours of data in a dataframe.

It also has an automatic stopping mechanism that stops the code from running if
# of hours received is not equal to the desired/called hours of data.
(this will be edited to be more of a fluid process soon)
"""

#dependencies
from selenium import webdriver
from selenium.webdriver.common.by import By
import pandas as pd
from io import StringIO
import math
import sys
import requests_cache
from retry_requests import retry
from datetime import datetime
import pytz

#adjust station according to desired station; station ID found on wrh.noaa.gov/map/
#adjust hours as necessary
url = "https://www.weather.gov/wrh/timeseries"

#time calculation depending on current time to find # hours to call for web scrape (match index with API)
now = datetime.now(pytz.utc)
h = now.strftime("%H")
h = 23-int(float(h))

site = "K20V"
hours = 720-h

url = url + f"?site={site}&hours={hours}&hourly=True"

driver = webdriver.Chrome()
driver.get(url)

driver.implicitly_wait(5)

table = driver.find_element(By.XPATH,'//*[@id="OBS_DATA"]').get_attribute('outerHTML')
df_table = pd.read_html(StringIO(table))[0]

#flip index order of table
df_table = df_table.iloc[::-1]
print(df_table.head(3))
#print resulting table to confirm first timeframes of historical data

desired_length = hours

# Check if the length of the dataframe is equal to the desired length
if len(df_table) != desired_length:
    print(f"The length of the dataframe is not equal to {desired_length}. Stopping the process.")
    driver.close()
    sys.exit(1)
else:
    print("The length of the dataframe is equal to the desired length. Continuing with the process.")

#get lat, long, and elev data from station webpage
location = driver.find_element(By.XPATH,'//*[@id="SITE"]/p').get_attribute('outerHTML')

location_parse = location.split("<br>")
info = location_parse[2]
info = info.split(" ")
elev = math.ceil(float(info[1]))/3.28084 #3.28084 is factor to change feet -> meters
coord = info[4].split("/")
lat = float(coord[0])
long = float(coord[1])
station_name = location_parse[1].split("(")
station_name = station_name[0]

driver.close()

"""
PART 2: OPEN-METEO API

Open-Meteo provides an easy, free call system for forecasts (even historical forecasts)

The below script is their example script for calling from Python with slight
param and end edits.
"""

import openmeteo_requests

# Setup the Open-Meteo API client with cache and retry on error
cache_session = requests_cache.CachedSession('.cache', expire_after = 3600)
retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
openmeteo = openmeteo_requests.Client(session = retry_session)

# The order of variables in hourly or daily is important to assign them correctly below
url = "https://api.open-meteo.com/v1/forecast"
params = {
	"latitude": lat,
	"longitude": long,
    "elevation": elev,
	"hourly": ["temperature_2m", "dew_point_2m", "precipitation", "pressure_msl", "cloud_cover_low", "cloud_cover_mid", "cloud_cover_high", "wind_speed_10m", "wind_gusts_10m"],
	"temperature_unit": "fahrenheit",
	"wind_speed_unit": "mph",
	"precipitation_unit": "inch",
	"past_days": 30,
	"models": "best_match"
}
responses = openmeteo.weather_api(url, params=params)

# Process first location. Add a for-loop for multiple locations or weather models
response = responses[0]

# Process hourly data. The order of variables needs to be the same as requested.
hourly = response.Hourly()
hourly_temperature_2m = hourly.Variables(0).ValuesAsNumpy()
hourly_dew_point_2m = hourly.Variables(1).ValuesAsNumpy()
hourly_precipitation = hourly.Variables(2).ValuesAsNumpy()
hourly_pressure_msl = hourly.Variables(3).ValuesAsNumpy()
hourly_cloud_cover_low = hourly.Variables(4).ValuesAsNumpy()
hourly_cloud_cover_mid = hourly.Variables(5).ValuesAsNumpy()
hourly_cloud_cover_high = hourly.Variables(6).ValuesAsNumpy()
hourly_wind_speed_10m = hourly.Variables(7).ValuesAsNumpy()
hourly_wind_gusts_10m = hourly.Variables(8).ValuesAsNumpy()

hourly_data = {"date": pd.date_range(
	start = pd.to_datetime(hourly.Time(), unit = "s"),
	end = pd.to_datetime(hourly.TimeEnd(), unit = "s"),
	freq = pd.Timedelta(seconds = hourly.Interval()),
	inclusive = "left"
)}
hourly_data["temperature_2m"] = hourly_temperature_2m
hourly_data["dew_point_2m"] = hourly_dew_point_2m
hourly_data["precipitation"] = hourly_precipitation
hourly_data["pressure_msl"] = hourly_pressure_msl
hourly_data["cloud_cover_low"] = hourly_cloud_cover_low
hourly_data["cloud_cover_mid"] = hourly_cloud_cover_mid
hourly_data["cloud_cover_high"] = hourly_cloud_cover_high
hourly_data["wind_speed_10m"] = hourly_wind_speed_10m
hourly_data["wind_gusts_10m"] = hourly_wind_gusts_10m

hourly_dataframe = pd.DataFrame(data = hourly_data)
print(hourly_dataframe.head(3))
#make sure match index with retrieved historical data (UTC time)

hourly_data = hourly_dataframe.drop(['date'],axis=1)

"""
PART 3: Data Processing

This section makes sure to concat both of the dataframes as necessary for
matching index values across data retrieved.
"""

import numpy as np

def justify(a, invalid_val=0, axis=1, side='left'):    
    """
    Justifies a 2D array

    Parameters
    ----------
    A : ndarray
        Input array to be justified
    axis : int
        Axis along which justification is to be made
    side : str
        Direction of justification. It could be 'left', 'right', 'up', 'down'
        It should be 'left' or 'right' for axis=1 and 'up' or 'down' for axis=0.

    """

    if invalid_val is np.nan:
        #change to notnull
        mask = pd.notnull(a)
    else:
        mask = a!=invalid_val
    justified_mask = np.sort(mask,axis=axis)
    if (side=='up') | (side=='left'):
        justified_mask = np.flip(justified_mask,axis=axis)
    #change dtype to object
    out = np.full(a.shape, invalid_val, dtype=object)  
    if axis==1:
        out[justified_mask] = a[mask]
    else:
        out.T[justified_mask.T] = a.T[mask.T]
    return out

#process concat data
df = pd.concat([df_table, hourly_data], axis=1)
arr = justify(df.to_numpy(), invalid_val=np.nan,axis=0)
df = pd.DataFrame(arr, columns=df.columns, index=df.index)
df = df.reset_index()
df = df.drop(['index'], axis=1)
df = df.iloc[:int(float(len(hourly_dataframe)))]

"""
PART 4: ARIMA Modeling

This section runs the data that was gathered through the first three steps into
an ARIMA (Auto(R)egressive Integrated Moving Average) model .

This model finds a best fit for the data retrieved from the previous steps and
outputs a forecast, which is also graphed in pyplot and visible in plot explorers.
"""

import matplotlib.pyplot as plt
from skforecast.Sarimax import Sarimax
import warnings
warnings.filterwarnings('ignore')

#number equals # hours in beginning url
df1 = df.iloc[:(int(float(len(df_table))))]
df2 = df.iloc[(int(float(len(df_table)))):]
temp = df1.columns[1]

#customize for better results
arima = Sarimax(order=(72, 1, 24))
arima.fit(y=df1[temp].astype(float),exog=df1[['temperature_2m','dew_point_2m','precipitation','pressure_msl','cloud_cover_low','cloud_cover_mid','cloud_cover_high','wind_speed_10m','wind_gusts_10m']].astype(float))

#customize steps number on how far ahead want to forecast
predictions = arima.predict(steps=120,exog=df2[['temperature_2m','dew_point_2m','precipitation','pressure_msl','cloud_cover_low','cloud_cover_mid','cloud_cover_high','wind_speed_10m','wind_gusts_10m']].astype(float))

#plot forecast
fig, ax = plt.subplots(figsize=(20,8))
df[temp].plot(ax=ax, label='Actual')
predictions.plot(ax=ax, label='Forecast')
plt.grid(True)
plt.title(f'{station_name} Temperature Forecast')
ax.legend()
