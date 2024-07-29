# How to use config files

**Generate training script**

From the `miles-credit` base folder:
```
python applications/train.py -c config/fuxi_baseline_ksha_static.yml -l 1
```

**Run inference**

From `miles-credit/applications` folder:
```
python predict.py -c ../results/fuxi/model.yml
```

# Config file structure break down

```
save_loc: "/glade/work/$USER/"
```
`save_loc`: the file path where model configurations, training scripts, and model weights are stored; it supports environment variables, such as `$USER`. If `save_loc` does not exist, it will be created automatically.

## data section

```
variables: ['U','V','T','Q']
surface_variables: ['SP','t2m','V500','U500','T500','Z500','Q500']
static_variables: ['Z_GDS4_SFC','LSM','tsi']
save_loc: '/glade/derecho/scratch/schreck/STAGING/TOTAL_*'
TOA_forcing_path: '/glade/derecho/scratch/dgagne/credit_scalers/solar_radiation_2024-03-29_0204.nc' 
```

`variables`: upper air atmospheric variables that the model predicts.

`surface_variables`: surface or diagnostiv variables that the model predicts.

`save_loc`: locations of training and validation data. It is organized as yearly `zarr` files and will be improted using `xarray`.
* The name of `variables` and `surface_variables` must match the variable keys of the `zarr` files 
* Training, validation, and testing set split can be specified in `miles-credit/applications/train.py`
      
`static_variables`: static fields that are used as model inputs; they will not be predicted
* `Z_GDS4_SFC`: Geopotential relative to the mean sea level [Link](https://codes.ecmwf.int/grib/param-db/129)
* `LSM`: Land-sea mask [Link](https://codes.ecmwf.int/grib/param-db/172)
* `tsi`: Total solar irradiance
* `Z_GDS4_SFC` and `LSM` are stored in `latitude_weights` in the [loss section](<loss section>). `tsi` is stored in the `TOA_forcing_path`

```
history_len: 2 
forecast_len: 0
valid_history_len: 2
valid_forecast_len: 0

```  
`history_len` and `valid_history_len`: Number of previous forecast lead times as inputs.

`forecast_len` and `forecast_len`: Number of forecast lead times to predict. Zero means the next lead time only.

## trainer section

```
mode: ddp # or none or fsdp
```

`mode`: training and inference strategies. `none` = no GPU usage; `ddp` = Distributed Data Parallel; `fsdp` = Fully Sharded Data Parallel

```
train_batch_size: 2
valid_batch_size: 2
```

`train_batch_size` and `valid_batch_size`: number of batches per GPU, e.g., `train_batch_size: 2` with 32 GPUs mean batch size = 64.

## model section

```
type: "crossformer" # or "fuxi" or "unet"
```

`type`: The name of the AI weather prediction model. Currently supports Crossformer, FuXi, and U-net; it will be expanded and updated.

```
channels: 4          # Channels (default: 4)
surface_channels: 7  # Surface channels (default: 7)
static_channels: 3   #
```

`channels`, `surface_channels`, `static_channels`: number of upper air, surface, and static variables. The numbers must be agreed with the [data section](<data section>).

## loss section

```
latitude_weights: "/glade/u/home/wchapman/MLWPS/DataLoader/LSM_static_variables_ERA5_zhght.nc"
```

`latitude_weights`: a netCDF file that contains geographical information. It is required to have the following variables:
* `latitude`, `longitude`: 1-D latitude and longitude coordinates.
* (optionally) `Z_GDS4_SFC`, `LSM`: static variables as in the [data section](<data section>).

## predict section

```
forecasts: [
    ["2018-06-01 00:00:00", "2018-06-01 02:00:00"]
]
use_laplace_filter: True
save_format: "nc"
```

`forecasts` the range of initialization and forecast-end times. It is formated as follows:

```
[
  [initialization time, forecast-end time],
  [initialization time, forecast-end time],
]
```

Forecast lead time is speccified in `forecast_len` in the [data section](<data section>).




