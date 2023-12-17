from selenium import webdriver
from selenium.webdriver.common.by import By
import pandas as pd
from io import StringIO
import math

#adjust station according to desired station; station ID found on wrh.noaa.gov/map/
#adjust hours as necessary to match df indices
url = "https://www.weather.gov/wrh/timeseries?site=HRBC2&hours=472&hourly=true"

driver = webdriver.Chrome()
driver.get(url)

#let webpage load table
driver.implicitly_wait(5)

print("Driver loaded successfully!")

table = driver.find_element(By.XPATH,'//*[@id="OBS_DATA"]').get_attribute('outerHTML')
df = pd.read_html(StringIO(table))[0]

#flip index order of table
df = df.iloc[::-1]

#get lat, long, and elev data from station webpage
location = driver.find_element(By.XPATH,'//*[@id="SITE"]/p').get_attribute('outerHTML')

location_parse = location.split(" ")
elev = location_parse[10]
elev = math.ceil(float(elev))/3.28084
coord = location_parse[13]
coord = coord.split("/")
lat = float(coord[0])
long = coord[1]
long = long.split("<")
long = float(long[0])

driver.close()

import openmeteo_requests
import requests_cache
from retry_requests import retry

# Setup the Open-Meteo API client with cache and retry on error
cache_session = requests_cache.CachedSession('.cache', expire_after = 3600)
retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
openmeteo = openmeteo_requests.Client(session = retry_session)

# Make sure all required weather variables are listed here
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
	"timezone": "auto",
	"past_days": 19,
	"models": "best_match"
}
responses = openmeteo.weather_api(url, params=params)

# Process first location. Add a for-loop for multiple locations or weather models
response = responses[0]
print(f"Coordinates {response.Latitude()}°E {response.Longitude()}°N")
print(f"Elevation {response.Elevation()} m asl")
print(f"Timezone {response.Timezone()} {response.TimezoneAbbreviation()}")
print(f"Timezone difference to GMT+0 {response.UtcOffsetSeconds()} s")

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
print(hourly_dataframe)

hourly_data = hourly_dataframe.drop(['date'],axis=1)

print("Historical weather forecast information processed!")

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
df = pd.concat([df, hourly_data], axis=1)
arr = justify(df.to_numpy(), invalid_val=np.nan,axis=0)
df = pd.DataFrame(arr, columns=df.columns, index=df.index)
df = df.reset_index()
df = df.drop(['index'], axis=1)
df = df.iloc[:648]

print("Dataframe setup complete!")

import matplotlib.pyplot as plt
from skforecast.Sarimax import Sarimax
import warnings
warnings.filterwarnings('ignore')

#number equals # hours (minus 1) in beginning url
df1 = df.iloc[:471]
df2 = df.iloc[471:]
temp = df1.columns[1]

#customize for better results
arima = Sarimax(order=(48, 1, 12))
arima.fit(y=df1[temp].astype(float),exog=df1[['temperature_2m','dew_point_2m','precipitation','pressure_msl','cloud_cover_low','cloud_cover_mid','cloud_cover_high','wind_speed_10m','wind_gusts_10m']].astype(float))
arima.summary()

print("ARIMA setup!")

#customize steps number on how far ahead want to forecast
predictions = arima.predict(steps=48,exog=df2[['temperature_2m','dew_point_2m','precipitation','pressure_msl','cloud_cover_low','cloud_cover_mid','cloud_cover_high','wind_speed_10m','wind_gusts_10m']].astype(float))

#plot forecast
fig, ax = plt.subplots(figsize=(20,8))
df[temp].plot(ax=ax, label='Actual')
predictions.plot(ax=ax, label='Forecast')
plt.grid(True)
ax.legend()
