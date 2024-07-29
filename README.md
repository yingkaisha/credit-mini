# NSF NCAR MILES Community Runnable Earth Digital Intelligence Twin (CREDIT) - Experimental & Research version

This repository is my personal version of CREDIT, diverged from the main CREDIT repository under the NSF NCAR GitHub Organization. It aims to provide a fast and light-weight computational environment for AI weather prediciton models. New features experimented within this repository may be forwarded to the main CREDIT. This repository will also updated with my own AI-weather-forecast-related research works. 

## Installation

This repository is primarily hosted on NSF NCAR HPCs (`casper.ucar.edu` and `derecho.hpc.ucar.edu`). Clone this repository, make sure you have Pytorch with GPU access, and go through the following steps:

```
mamba env create -f environment.yml
conda activate credit_mini
pip install .
```

## To Do

### 2024-07-26

* Switch from the current ERA5 training set to the model resolution / level ERA5 from Google.
* Add more forcing and static inputs.
* Base class and unit testing.

# Contact
Yingkai Sha <ksha@ucar.edu>

