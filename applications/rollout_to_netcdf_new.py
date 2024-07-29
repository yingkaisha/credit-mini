# ---------- #
# System
import os
import gc
import sys
import yaml
import logging
import warnings
#import traceback
from glob import glob
from pathlib import Path
from argparse import ArgumentParser
import multiprocessing as mp

# ---------- #
# Numerics
from datetime import datetime, timedelta
import pandas as pd
import xarray as xr
import numpy as np

# ---------- #
# AI libs
import torch
import torch.distributed as dist
from torchvision import transforms
# import wandb

# ---------- #
# credit
from credit.models import load_model
from credit.seed import seed_everything
from credit.data import Predict_Dataset, concat_and_reshape, reshape_only
from credit.transforms import load_transforms, Normalize_ERA5_and_Forcing
from credit.pbs import launch_script, launch_script_mpi
from credit.pol_lapdiff_filt import Diffusion_and_Pole_Filter
from credit.forecast import load_forecasts
from credit.distributed import distributed_model_wrapper
from credit.models.checkpoint import load_model_state
from credit.output import load_metadata, make_xarray, save_netcdf_increment
from torch.utils.data import get_worker_info
from torch.utils.data.distributed import DistributedSampler


logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

def setup(rank, world_size, mode):
    logging.info(f"Running {mode.upper()} on rank {rank} with world_size {world_size}.")
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def predict(rank, world_size, conf, p):

    # setup rank and world size for GPU-based rollout
    if conf["trainer"]["mode"] in ["fsdp", "ddp"]:
        setup(rank, world_size, conf["trainer"]["mode"])

    # infer device id from rank
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{rank % torch.cuda.device_count()}")
        torch.cuda.set_device(rank % torch.cuda.device_count())
    else:
        device = torch.device("cpu")

    # config settings
    seed = 1000 if "seed" not in conf else conf["seed"]
    seed_everything(seed)

    # number of input time frames 
    history_len = conf["data"]["history_len"]

    # transform and ToTensor class
    transform = load_transforms(conf)
    if conf["data"]["scaler_type"] == 'std_new':
        state_transformer = Normalize_ERA5_and_Forcing(conf)
    else:
        print('Scaler type {} not supported'.format(conf["data"]["scaler_type"]))
        raise
    # ----------------------------------------------------------------- #
    # parse varnames and save_locs from config
    if 'lead_time_periods' in conf['data']:
        lead_time_periods = conf['data']['lead_time_periods']
    else:
        lead_time_periods = 1
    
    ## upper air variables
    all_ERA_files = sorted(glob(conf["data"]["save_loc"]))
    varname_upper_air = conf['data']['variables']
    
    ## surface variables
    if "save_loc_surface" in conf["data"]:
        surface_files = sorted(glob(conf["data"]["save_loc_surface"]))
        varname_surface = conf['data']['surface_variables']
    else:
        surface_files = None
        varname_surface = None 
    
    ## forcing variables
    if ('forcing_variables' in conf['data']) and (len(conf['data']['forcing_variables']) > 0):
        forcing_files = conf['data']['save_loc_forcing']
        varname_forcing = conf['data']['forcing_variables']
    else:
        forcing_files = None
        varname_forcing = None
    
    ## static variables
    if ('static_variables' in conf['data']) and (len(conf['data']['static_variables']) > 0):
        static_files = conf['data']['save_loc_static']
        varname_static = conf['data']['static_variables']
    else:
        static_files = None
        varname_static = None

    # ----------------------------------------------------------------- #\
    # get dataset
    dataset = Predict_Dataset(
        conf, 
        varname_upper_air,
        varname_surface,
        varname_forcing,
        varname_static,
        filenames=all_ERA_files,
        filename_surface=surface_files,
        filename_forcing=forcing_files,
        filename_static=static_files,
        fcst_datetime=load_forecasts(conf),
        history_len=history_len,
        rank=rank,
        world_size=world_size,
        transform=transform,
        rollout_p=0.0,
        which_forecast=None
    )
    # setup the dataloder
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        pin_memory=True,
        num_workers=0,
        drop_last=False,
    )

    # load model
    model = load_model(conf, load_weights=True).to(device)

    # Warning -- see next line
    distributed = conf["trainer"]["mode"] in ["ddp", "fsdp"]
    if distributed:  # A new field needs to be added to predict
        model = distributed_model_wrapper(conf, model, device)
        if conf["trainer"]["mode"] == "fsdp":
            # Load model weights (if any), an optimizer, scheduler, and gradient scaler
            model = load_model_state(conf, model, device)

    model.eval()

    # get lat/lons from x-array
    latlons = xr.open_dataset(conf["loss"]["latitude_weights"])

    meta_data = load_metadata(conf)

    # Set up the diffusion and pole filters
    if (
        "use_laplace_filter" in conf["predict"]
        and conf["predict"]["use_laplace_filter"]
    ):
        dpf = Diffusion_and_Pole_Filter(
            nlat=conf["model"]["image_height"],
            nlon=conf["model"]["image_width"],
            device=device,
        )

    # Rollout
    with torch.no_grad():
        # forecast count = a constant for each run
        forecast_count = 0
    
        # y_pred allocation
        y_pred = None
        static = None
        results = []
    
        # model inference loop
        for k, batch in enumerate(data_loader):
    
            # get the datetime and forecasted hours
            date_time = batch["datetime"].item()
            forecast_hour = batch["forecast_hour"].item()
            # initialization on the first forecast hour
            if forecast_hour == 1:
                
                # Initialize x and x_surf with the first time step
                if "x_surf" in batch:
                    # combine x and x_surf
                    # input: (batch_num, time, var, level, lat, lon), (batch_num, time, var, lat, lon) 
                    # output: (batch_num, var, time, lat, lon), 'x' first and then 'x_surf'
                    x = concat_and_reshape(batch["x"], batch["x_surf"]).to(device).float()
                else:
                    # no x_surf
                    x = reshape_only(batch["x"]).to(device).float()

                init_datetime_str = datetime.utcfromtimestamp(date_time)
                init_datetime_str = init_datetime_str.strftime('%Y-%m-%dT%HZ')

            # -------------------------------------------------------------------------------------- #
            # add forcing and static variables (regardless of fcst hours)
            if 'x_forcing_static' in batch:
                
                # (batch_num, time, var, lat, lon) --> (batch_num, var, time, lat, lon)
                x_forcing_batch = batch['x_forcing_static'].to(device).permute(0, 2, 1, 3, 4).float()

                # concat on var dimension
                x = torch.cat((x, x_forcing_batch), dim=1)

            # -------------------------------------------------------------------------------------- #
            # start prediction
            y_pred = model(x)
            y_pred = state_transformer.inverse_transform(y_pred.cpu())
            
            if ("use_laplace_filter" in conf["predict"] and conf["predict"]["use_laplace_filter"]):
                y_pred = (
                    dpf.diff_lap2d_filt(y_pred.to(device).squeeze())
                    .unsqueeze(0)
                    .unsqueeze(2)
                    .cpu()
                )
    
            # Save the current forecast hour data in parallel
            utc_datetime = datetime.utcfromtimestamp(date_time) + timedelta(hours=lead_time_periods*forecast_hour)
    
            # convert the current step result as x-array
            darray_upper_air, darray_single_level = make_xarray(
                y_pred,
                utc_datetime,
                latlons.latitude.values,
                latlons.longitude.values,
                conf,
            )
            
            # Save the current forecast hour data in parallel
            result = p.apply_async(
                save_netcdf_increment,
                (
                    darray_upper_air, 
                     darray_single_level, 
                     init_datetime_str, 
                     lead_time_periods*forecast_hour, 
                     meta_data, 
                     conf
                )
            )
            results.append(result)
            
            # Update the input
            # setup for next iteration, transform to z-space and send to device
            y_pred = state_transformer.transform_array(y_pred).to(device)
    
            if history_len == 1:
                x = y_pred.detach()
            else:
                # use multiple past forecast steps as inputs
                # static channels will get updated on next pass
                static_dim_size = abs(x.shape[1] - y_pred.shape[1])
                
                # if static_dim_size=0 then :0 gives empty range
                x_detach = x[:, :-static_dim_size, 1:].detach() if static_dim_size else x[:, :, 1:].detach()  
                x = torch.cat([x_detach, y_pred.detach()], dim=2)
    
            # Explicitly release GPU memory
            torch.cuda.empty_cache()
            gc.collect()
    
            if batch["stop_forecast"][0]:
                # Wait for all processes to finish in order
                for result in results:
                    result.get()
    
                # Now merge all the files into one and delete leftovers
                # merge_netcdf_files(init_datetime_str, conf)
    
                # forecast count = a constant for each run
                forecast_count += 1
    
                # update lists
                results = []
    
                # y_pred allocation
                y_pred = None
    
                gc.collect()
    
                if distributed:
                    torch.distributed.barrier()
    
    if distributed:
        torch.distributed.barrier()

    return 1


if __name__ == "__main__":

    description = "Rollout AI-NWP forecasts"
    parser = ArgumentParser(description=description)
    # -------------------- #
    # parser args: -c, -l, -w
    parser.add_argument(
        "-c",
        dest="model_config",
        type=str,
        default=False,
        help="Path to the model configuration (yml) containing your inputs.",
    )

    parser.add_argument(
        "-l",
        dest="launch",
        type=int,
        default=0,
        help="Submit workers to PBS.",
    )

    parser.add_argument(
        "-w",
        "--world-size",
        type=int,
        default=4,
        help="Number of processes (world size) for multiprocessing",
    )

    parser.add_argument(
        "-m",
        "--mode",
        type=str,
        default=0,
        help="Update the config to use none, DDP, or FSDP",
    )

    parser.add_argument(
        "-nd",
        "--no-data",
        type=str,
        default=0,
        help="If set to True, only pandas CSV files will we saved for each forecast",
    )
    parser.add_argument(
        "-s",
        "--subset",
        type=int,
        default=False,
        help="Predict on subset X of forecasts",
    )
    parser.add_argument(
        "-ns",
        "--no_subset",
        type=int,
        default=False,
        help="Break the forecasts list into X subsets to be processed by X GPUs",
    )
    parser.add_argument(
        "-cpus",
        "--num_cpus",
        type=int,
        default=8,
        help="Number of CPU workers to use per GPU",
    )

    # parse
    args = parser.parse_args()
    args_dict = vars(args)
    config = args_dict.pop("model_config")
    launch = int(args_dict.pop("launch"))
    mode = str(args_dict.pop("mode"))
    no_data = 0 if "no-data" not in args_dict else int(args_dict.pop("no-data"))
    subset = int(args_dict.pop("subset"))
    number_of_subsets = int(args_dict.pop("no_subset"))
    num_cpus = int(args_dict.pop("num_cpus"))

    # Set up logger to print stuff
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(levelname)s:%(name)s:%(message)s")

    # Stream output to stdout
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    root.addHandler(ch)

    # Load the configuration and get the relevant variables
    with open(config) as cf:
        conf = yaml.load(cf, Loader=yaml.FullLoader)
        
    # create a save location for rollout
    # ---------------------------------------------------- #
    assert 'save_forecast' in conf['predict'], "Please specify the output dir through conf['predict']['save_forecast']"
    
    forecast_save_loc = conf['predict']['save_forecast']
    os.makedirs(forecast_save_loc, exist_ok=True)
    
    print('Save roll-outs to {}'.format(forecast_save_loc))

    # Create a project directory (to save launch.sh and model.yml) if they do not exist
    save_loc = os.path.expandvars(conf["save_loc"])
    os.makedirs(save_loc, exist_ok=True)
    
    # Update config using override options
    if mode in ["none", "ddp", "fsdp"]:
        logger.info(f"Setting the running mode to {mode}")
        conf["trainer"]["mode"] = mode

    # Launch PBS jobs
    if launch:
        # Where does this script live?
        script_path = Path(__file__).absolute()
        if conf["pbs"]["queue"] == "casper":
            logging.info("Launching to PBS on Casper")
            launch_script(config, script_path)
        else:
            logging.info("Launching to PBS on Derecho")
            launch_script_mpi(config, script_path)
        sys.exit()

    #     wandb.init(
    #         # set the wandb project where this run will be logged
    #         project="Derecho parallelism",
    #         name=f"Worker {os.environ["RANK"]} {os.environ["WORLD_SIZE"]}"
    #         # track hyperparameters and run metadata
    #         config=conf
    #     )

    if number_of_subsets > 0:
        forecasts = load_forecasts(conf)
        if number_of_subsets > 0 and subset >= 0:
            subsets = np.array_split(forecasts, number_of_subsets)
            forecasts = subsets[subset - 1]  # Select the subset based on subset_size
            conf["predict"]["forecasts"] = forecasts

    seed = 1000 if "seed" not in conf else conf["seed"]
    seed_everything(seed)

    with mp.Pool(num_cpus) as p:
        if conf["trainer"]["mode"] in ["fsdp", "ddp"]:  # multi-gpu inference
            _ = predict(int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"]), conf, p=p)
        else:  # single device inference
            _ = predict(0, 1, conf, p=p)

    # Ensure all processes are finished
    p.close()
    p.join()
