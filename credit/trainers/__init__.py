import copy
import logging

# Import trainer classes
from credit.trainers.trainerERA5 import Trainer as TrainerERA5
from credit.trainers.trainerERA5_Diffusion import Trainer as TrainerERA5_Diffusion
from credit.trainers.trainerERA5_ensemble import Trainer as TrainerEnsemble
from credit.trainers.trainer404 import Trainer as Trainer404
from credit.trainers.ic_optimization import Trainer as TrainerIC

from credit.trainers.trainerLES import Trainer as TrainerLES
from credit.trainers.trainerWRF import Trainer as TrainerWRF
from credit.trainers.trainerWRF_multi import Trainer as TrainerWRF_Multi
from credit.trainers.trainerDscale import Trainer as TrainerDscale
from credit.trainers.trainerDiag import Trainer as TrainerDiag

logger = logging.getLogger(__name__)

# define trainer types
trainer_types = {
    "era5": (TrainerERA5, "Loading a single or multi-step trainer for the ERA5 dataset"),
    "era5-diffusion": (TrainerERA5_Diffusion, "Loading a single or multi-step diffusion trainer for the ERA5 dataset"),
    "era5-ensemble": (TrainerEnsemble, "Loading a single or multi-step trainer for the ERA5 dataset using CRPS loss"),
    "cam": (TrainerERA5, "Loading a single or multi-step trainer for the CAM dataset"),
    "ic-opt": (TrainerIC, "Loading an initial condition optimizer training class"),
    "conus404": (Trainer404, "Loading a standard trainer for the CONUS404 dataset."),
    "standard-les": (TrainerLES, "Loading a single-step LES trainer"),
    "standard-wrf": (TrainerWRF, "Loading a single-step WRF trainer"),
    "multi-step-wrf": (TrainerWRF_Multi, "Loading a multi-step WRF trainer"),
    'standard-dscale': (TrainerDscale, "Loading a downscaling trainer"),
    'standard-diag': (TrainerDiag, "Loading a diagnostic model trainer"),
}

def load_trainer(conf, load_weights=False):
    conf = copy.deepcopy(conf)
    trainer_conf = conf["trainer"]

    if "type" not in trainer_conf:
        msg = f"You need to specify a trainer 'type' in the config file. Choose from {list(trainer_types.keys())}"
        logger.warning(msg)
        raise ValueError(msg)

    trainer_type = trainer_conf.pop("type")

    if trainer_type in trainer_types:
        trainer, message = trainer_types[trainer_type]
        logger.info(message)
        return trainer
    else:
        msg = f"Trainer type {trainer_type} not supported. Exiting."
        logger.warning(msg)
        raise ValueError(msg)
