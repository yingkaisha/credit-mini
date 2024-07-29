import numpy as np
import pandas as pd
from pvlib.solarposition import get_solarposition
from pvlib.irradiance import get_extra_radiation
import xarray as xr
from tqdm import tqdm
import torch

class TOADataLoader:
    def __init__(self, conf):
        self.TOA = xr.open_dataset(conf["data"]["TOA_forcing_path"]).load()
        self.times_b = pd.to_datetime(self.TOA.time.values)

        # Precompute day of year and hour arrays
        self.days_of_year = self.times_b.dayofyear
        self.hours_of_day = self.times_b.hour

    def __call__(self, datetime_input):
        doy = datetime_input.dayofyear
        hod = datetime_input.hour

        # Use vectorized comparison for masking
        mask_toa = (self.days_of_year == doy) & (self.hours_of_day == hod)
        selected_tsi = self.TOA['tsi'].sel(time=mask_toa) / 2540585.74

        # Convert to tensor and add dimension
        return torch.tensor(selected_tsi.to_numpy()).unsqueeze(0).float()
        


def get_solar_radiation_loc(lon, lat, altitude, start_date, end_date, step_freq="1h", sub_freq="5Min"):
    """
    Calculate total solar irradiance at a single location over a range of times. Solar irradiance is integrated
    over the step frequency at specified substeps.

    Args:
        lon (float): longitude.
        lat (float): latitude.
        altitude (float): altitude in meters.
        start_date (str): date str for the beginning of the period (inclusive).
        end_date (str):  date str for the end of the period (inclusive).
        step_freq (str): period over which irradiance is integrated. Defaults to "1h".
        sub_freq (str): sub step frequency over the step period. Defaults to "5Min".

    Returns:
        xarray.DataArray: total solar irradiance time series with metadata.
    """
    start_date_ts = pd.Timestamp(start_date)
    end_date_ts = pd.Timestamp(end_date)
    step_sec = pd.Timedelta(step_freq).total_seconds()
    sub_sec = pd.Timedelta(sub_freq).total_seconds()
    step_len = int(step_sec // sub_sec)
    dates = pd.date_range(start=start_date_ts - pd.Timedelta(step_freq) + pd.Timedelta(sub_freq),
                          end=end_date_ts, freq=sub_freq)
    total_rad = get_extra_radiation(dates)
    solar_pos = get_solarposition(dates, lat, lon, altitude, method="nrel_numba")
    solar_rad = np.maximum(0, np.sin(np.radians(solar_pos["elevation"].values))) * total_rad
    step_rad = np.trapz(np.reshape(solar_rad, (int(solar_rad.size // step_len), step_len)),
                        dx=sub_sec, axis=1)
    step_dates = pd.date_range(start=start_date_ts, end=end_date_ts, freq=step_freq)
    out_rad_da = xr.DataArray(step_rad.reshape(-1, 1, 1),
                              coords={"time": step_dates, "latitude": [lat], "longitude": [lon]},
                              dims=("time", "latitude", "longitude"), name="tsi",
                              attrs={"long_name": "total solar irradiance", "units": "J m-2"})
    return out_rad_da


def get_solar_index(curr_date, ref_date="2000-01-01"):
    curr_date_ts = pd.to_datetime(curr_date)
    year_start = pd.Timestamp(f"{curr_date_ts.year:d}-01-01")
    curr_diff = curr_date_ts - year_start
    return int(curr_diff.total_seconds() / 3600)


if __name__ == "__main__":
    lons = np.arange(-100.0, -89.5, 0.5)
    lats = np.arange(30.0, 35.0, 0.5)
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    solar_ts = []
    for lon, lat in tqdm(zip(lon_grid.ravel(), lat_grid.ravel())):
        out = get_solar_radiation_loc(lon, lat, 0.0, "2016-01-01", "2016-12-31 23:00")
        solar_ts.append(out)
    combined = xr.combine_by_coords(solar_ts)
    print(combined)
