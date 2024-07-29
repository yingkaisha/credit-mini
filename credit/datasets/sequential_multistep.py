from credit.data import ERA5Dataset, find_key_for_number
from torch.utils.data import get_worker_info
from torch.utils.data.distributed import DistributedSampler
import xarray as xr
import torch


class DistributedSequentialDataset(torch.utils.data.IterableDataset):
    # https://colab.research.google.com/drive/1OFLZnX9y5QUFNONuvFsxOizq4M-tFvk-?usp=sharing#scrollTo=CxSCQPOMHgwo

    def __init__(self, filenames, history_len, forecast_len, skip_periods, rank, world_size, shuffle=False,
                 transform=None, rollout_p=0.0):

        self.dataset = ERA5Dataset(
            filenames=filenames,
            history_len=history_len,
            forecast_len=forecast_len,
            skip_periods=skip_periods,
            transform=transform
        )
        self.meta_data_dict = self.dataset.meta_data_dict
        self.all_fils = self.dataset.all_fils
        self.history_len = history_len
        self.forecast_len = forecast_len
        self.filenames = filenames
        self.transform = transform
        self.rank = rank
        self.world_size = world_size
        self.shuffle = shuffle
        self.skip_periods = skip_periods
        self.current_epoch = 0
        self.rollout_p = rollout_p

    def __len__(self):
        tlen = 0
        for bb in self.all_fils:
            tlen += (len(bb['time']) - self.forecast_len)
        return tlen

    def set_epoch(self, epoch):
        self.current_epoch = epoch

    def __iter__(self):
        worker_info = get_worker_info()
        num_workers = worker_info.num_workers if worker_info is not None else 1
        worker_id = worker_info.id if worker_info is not None else 0
        sampler = DistributedSampler(self, num_replicas=num_workers * self.world_size,
                                     rank=self.rank * num_workers + worker_id, shuffle=self.shuffle)
        sampler.set_epoch(self.current_epoch)

        for index in iter(sampler):
            result_key = find_key_for_number(index, self.meta_data_dict)
            true_ind = index - self.meta_data_dict[result_key][1]

            if true_ind > (len(self.all_fils[int(result_key)]['time']) - (self.history_len + self.forecast_len + 1)):
                true_ind = len(self.all_fils[int(result_key)]['time']) - (self.history_len + self.forecast_len + 3)

            indices = list(range(true_ind, true_ind + self.history_len + self.forecast_len))
            stop_forecast = False

            for k, ind in enumerate(indices):

                concatenated_samples = {'x': [], 'x_surf': [], 'y': [], 'y_surf': [], "static": [], "TOA": []}
                sliced = xr.open_zarr(self.filenames[int(result_key)], consolidated=True).isel(
                    time=slice(ind, ind + self.history_len + self.forecast_len + 1, self.skip_periods))

                historical_data = sliced.isel(time=slice(0, self.history_len)).load()
                target_data = sliced.isel(time=slice(self.history_len, self.history_len + 1)).load()

                sample = {
                    "x": historical_data,
                    "y": target_data,
                    "t": [
                        int(historical_data.time.values[0].astype('datetime64[s]').astype(int)),
                        int(target_data.time.values[0].astype('datetime64[s]').astype(int))
                    ]
                }

                if self.transform:
                    sample = self.transform(sample)

                for key in concatenated_samples.keys():
                    concatenated_samples[key] = sample[key].squeeze()

                stop_forecast = (k == self.forecast_len)

                concatenated_samples['forecast_hour'] = k
                concatenated_samples['index'] = index
                concatenated_samples['stop_forecast'] = stop_forecast
                concatenated_samples["datetime"] = [
                    int(historical_data.time.values[0].astype('datetime64[s]').astype(int)),
                    int(target_data.time.values[0].astype('datetime64[s]').astype(int))
                ]

                if self.history_len == 1:
                    concatenated_samples['x'] = concatenated_samples['x'].unsqueeze(0)
                    concatenated_samples['x_surf'] = concatenated_samples['x_surf'].unsqueeze(0)

                concatenated_samples['y'] = concatenated_samples['y'].unsqueeze(0)
                concatenated_samples['y_surf'] = concatenated_samples['y_surf'].unsqueeze(0)

                yield concatenated_samples

                if stop_forecast:
                    break

                if (k == self.forecast_len):
                    break
