import warnings
import os
import sys
import yaml
import wandb
import shutil
import logging

from pathlib import Path
from argparse import ArgumentParser

import torch
import torch.distributed as dist
from torchsummary import summary

from credit.models import load_model
from credit.pbs import launch_script, launch_script_mpi
from credit.seed import seed_everything
from credit.distributed import distributed_model_wrapper


warnings.filterwarnings("ignore")

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

# https://stackoverflow.com/questions/59129812/how-to-avoid-cuda-out-of-memory-in-pytorch
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


def setup(rank, world_size, mode):
    logging.info(f"Running {mode.upper()} on rank {rank} with world_size {world_size}.")
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


# def distributed_model_wrapper(conf, neural_network, device):

#     if conf["trainer"]["mode"] == "fsdp":

#         # Define the sharding policies

#         if "crossformer" in conf["model"]["type"]:
#             from credit.models.crossformer_skip import Attention as Attend
#         elif "fuxi" in conf["model"]["type"]:
#             from credit.models.fuxi import UTransformer as Attend
#         else:
#             raise OSError("You asked for FSDP but only crossformer and fuxi are currently supported.")

#         auto_wrap_policy1 = functools.partial(
#             transformer_auto_wrap_policy,
#             transformer_layer_cls={Attend}
#         )

#         auto_wrap_policy2 = functools.partial(
#             size_based_auto_wrap_policy, min_num_params=1_000
#         )

#         def combined_auto_wrap_policy(module, recurse, nonwrapped_numel):
#             # Define a new policy that combines policies
#             p1 = auto_wrap_policy1(module, recurse, nonwrapped_numel)
#             p2 = auto_wrap_policy2(module, recurse, nonwrapped_numel)
#             return p1 or p2

#         # Mixed precision

#         use_mixed_precision = conf["trainer"]["use_mixed_precision"] if "use_mixed_precision" in conf["trainer"] else False

#         logging.info(f"Using mixed_precision: {use_mixed_precision}")

#         if use_mixed_precision:
#             for key, val in conf["trainer"]["mixed_precision"].items():
#                 conf["trainer"]["mixed_precision"][key] = parse_dtype(val)
#             mixed_precision_policy = MixedPrecision(**conf["trainer"]["mixed_precision"])
#         else:
#             mixed_precision_policy = None

#         # CPU offloading

#         cpu_offload = conf["trainer"]["cpu_offload"] if "cpu_offload" in conf["trainer"] else False

#         logging.info(f"Using CPU offloading: {cpu_offload}")

#         # FSDP module

#         model = TorchFSDPModel(
#             neural_network,
#             use_orig_params=True,
#             auto_wrap_policy=combined_auto_wrap_policy,
#             mixed_precision=mixed_precision_policy,
#             cpu_offload=CPUOffload(offload_params=cpu_offload)
#         )

#         # activation checkpointing on the transformer blocks

#         activation_checkpoint = conf["trainer"]["activation_checkpoint"] if "activation_checkpoint" in conf["trainer"] else False

#         logging.info(f"Activation checkpointing: {activation_checkpoint}")

#         if activation_checkpoint:

#             # https://pytorch.org/blog/efficient-large-scale-training-with-pytorch/

#             non_reentrant_wrapper = functools.partial(
#                 checkpoint_wrapper,
#                 checkpoint_impl=CheckpointImpl.NO_REENTRANT,
#             )

#             check_fn = lambda submodule: isinstance(submodule, Attend)

#             apply_activation_checkpointing(
#                 model,
#                 checkpoint_wrapper_fn=non_reentrant_wrapper,
#                 check_fn=check_fn
#             )

#     elif conf["trainer"]["mode"] == "ddp":
#         model = DDP(neural_network, device_ids=[device])
#     else:
#         model = neural_network

#     return model


def main(rank, world_size, conf, trial=False):

    if conf["trainer"]["mode"] in ["fsdp", "ddp"]:
        setup(rank, world_size, conf["trainer"]["mode"])

    # Config settings

    channels = conf["model"]["levels"] * len(conf["data"]["variables"]) + len(conf["data"]["surface_variables"])
    if "static_variables" in conf["data"]:
        channels += len(conf["data"]["static_variables"])

    frames = conf["model"]["frames"]
    height = conf["model"]["image_height"]
    width = conf["model"]["image_width"]

    # Set device

    device = torch.device(f"cuda:{rank % torch.cuda.device_count()}") if torch.cuda.is_available() else torch.device("cpu")
    torch.cuda.set_device(rank % torch.cuda.device_count())

    # Load model

    m = load_model(conf)

    # send the module to the correct device first

    m.to(device)

    # Wrap if using DDP or FSDP

    model = distributed_model_wrapper(conf, m, device)

    try:
        summary(model, input_size=(channels, frames, height, width))
    except RuntimeError as e:
        if "CUDA" in str(e):
            logging.warning(f"CUDA out of memory error occurred: {e}.")
        else:
            logging.warning(f"An error occurred: {e}")


if __name__ == "__main__":

    description = "Train a segmengation model on a hologram data set"
    parser = ArgumentParser(description=description)
    parser.add_argument(
        "-c",
        "--config",
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
        "--wandb",
        dest="wandb",
        type=int,
        default=0,
        help="Use wandb. Default = False"
    )
    args = parser.parse_args()
    args_dict = vars(args)
    config = args_dict.pop("model_config")
    launch = int(args_dict.pop("launch"))
    use_wandb = int(args_dict.pop("wandb"))

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

    # Create directories if they do not exist and copy yml file
    save_loc = os.path.expandvars(conf["save_loc"])
    os.makedirs(save_loc, exist_ok=True)
    if not os.path.exists(os.path.join(save_loc, "model.yml")):
        shutil.copy(config, os.path.join(save_loc, "model.yml"))

    # Launch PBS jobs
    if launch:
        # Where does this script live?
        script_path = Path(__file__).absolute()
        if conf['pbs']['queue'] == 'casper':
            logging.info("Launching to PBS on Casper")
            launch_script(config, script_path)
        else:
            logging.info("Launching to PBS on Derecho")
            launch_script_mpi(config, script_path)
        sys.exit()

    if use_wandb:  # this needs updated
        wandb.init(
            # set the wandb project where this run will be logged
            project="Derecho parallelism",
            name=f"Worker {os.environ['RANK']} {os.environ['WORLD_SIZE']}",
            # track hyperparameters and run metadata
            config=conf
        )

    seed = 1000 if "seed" not in conf else conf["seed"]
    seed_everything(seed)

    if conf["trainer"]["mode"] in ["fsdp", "ddp"]:
        main(int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"]), conf)
    else:
        main(0, 1, conf)
