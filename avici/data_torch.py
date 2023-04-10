import multiprocessing
import time
import psutil
from collections import defaultdict
import functools

import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
from torch.utils.data import IterableDataset
from typing import Optional

from avici.definitions import RNG_ENTROPY_TRAIN
from avici.buffer import SharedState, get_soft_sample_insert_ratio, is_buffer_filled
from avici.utils.torch import structured_py_function
import pyarrow.plasma as plasma

def _device_put_sharded(sharded_tree, devices):
    """Taken from  https://github.com/deepmind/dm-haiku/blob/main/examples/imagenet/dataset.py """
    leaves, treedef = torch.utils.data.get_worker_info().dataset.tree_flatten(sharded_tree)
    n = leaves[0].shape[0]
    return torch.device_put_sharded([torch.tree_unflatten(treedef, [l[i] for l in leaves]) for i in range(n)],
                                  devices)

def _double_cache(ds):
    """Taken from https://github.com/deepmind/dm-haiku/blob/main/examples/imagenet/dataset.py.

    Keeps at least two batches on the accelerator.
    The current GPU allocator design reuses previous allocations. For a training
    loop this means batches will (typically) occupy the same region of memory as
    the previous batch. An issue with this is that it means we cannot overlap a
    host->device copy for the next batch until the previous step has finished and
    the previous batch has been freed.
    By double buffering we ensure that there are always two batches on the device.
    This means that a given batch waits on the N-2'th step to finish and free,
    meaning that it can allocate and copy the next batch to the accelerator in
    parallel with the N-1'th step being executed.
    Args:
        ds: Iterable of batches of numpy arrays.
    Yields:
        Batches of sharded device arrays.
    """
    batch = None
    devices = torch.cuda.device_count()
    for next_batch in ds:
        assert next_batch is not None
        next_batch = _device_put_sharded(next_batch, devices)
        if batch is not None:
            yield batch
        batch = next_batch
    if batch is not None:
        yield batch


import torch.multiprocessing as mp
from collections import defaultdict
import time
import psutil
import torch.utils.data as data_utils
import numpy as np

class SharedState:
    def __init__(self, queues, buffers, is_alive, next_job, plasma_store_name):
        self.queues = queues
        self.buffers = buffers
        self.is_alive = is_alive
        self.next_job = next_job
        self.plasma_store_name = plasma_store_name

def get_soft_sample_insert_ratio(buffer):
    return buffer.metrics.soft_sample_insert_ratio

def is_buffer_filled(buffer):
    return buffer.is_filled

def structured_torch_function(func, inp, Tout):
    out = func(*inp)
    if isinstance(out, tuple):
        return tuple(torch.as_tensor(x).to(torch.float32) for x in out)
    else:
        return torch.as_tensor(out).to(torch.float32)

class AsyncBufferDataset: 
    def __init__(self, *,
                 seed,
                 config,
                 buffer_class,
                 buffer_size,
                 train_n_observations_obs,
                 train_n_observations_int,
                 batch_dims_train=None,
                 batch_dims_test=None,
                 double_cache_train=True,
                 n_workers=None,
                 n_listeners=1,
                 verbose=True,
                 queue_max_size=100,
                 object_store_gb=10.0,
                 ):

        self.config = config
        self.buffer_class = buffer_class
        self.train_n_observations_obs = train_n_observations_obs
        self.train_n_observations_int = train_n_observations_int
        self.batch_dims_train = batch_dims_train
        self.batch_dims_test = batch_dims_test
        self.double_cache_train = double_cache_train
        self.verbose = verbose

        if self.verbose:
            print(f"AsyncBufferDataset", flush=True)

        # create plasma object store server
        # this emulates `with plasma.start_plasma_store as self._plasma_store: ...` and is closed in self.finish())
        self._plasma_store = plasma.start_plasma_store(int(object_store_gb * 1024 * 1024 * 1024))
        self._plasma_store_name, self._proc = type(self._plasma_store).__enter__(self._plasma_store)

        self._main_plasma_client = plasma.connect(self._plasma_store_name)

        # setup workers via async multiprocessing
        try:
            # only on linux but only correct call on cluster (psutil.cpu_count is false)
            cpu_count = len(psutil.Process().cpu_affinity())
        except AttributeError:
            cpu_count = psutil.cpu_count(logical=True)

        self.n_workers = n_workers or cpu_count
        self.n_listeners = n_listeners if self.n_workers >= 16 else 1 # with few CPUs don't need more than one listener

        n_processes = self.n_workers + self.n_listeners + 1
        self._pool = mp.Pool(processes=n_processes)
        self._endless_jobs = []
        self._listener_jobs = []
        if self.verbose:
            print(f"cpu count: {cpu_count}", flush=True)
            print(f"processes: {n_processes} "
                  f"({self.n_workers} workers,"
                  f" {self.n_listeners} listeners,"
                  f" {1} load balancer)", flush=True)

        # init shared state
        manager = mp.Manager()

        self._state = SharedState(
            queues=[manager.Queue(maxsize=queue_max_size) for _ in range(self.n_listeners)],
            buffers=defaultdict(dict),
        
    def _make_dataset(self, buffer, is_train: bool, batch_dims=None):
    """
    Launch consumer (buffer consumer) torch.utils.data.Dataset iterator to support batching, prefetching, etc.
    for buffers `descr`
    """

    if is_train:
        # plasma client
        client = plasma.connect(self._plasma_store_name)

        # infinite dataset that samples continually from updating buffer
        ds = torch.utils.data.IterableDataset()
        rng = onp.random.default_rng(self.ds_seed)

        def infinite_buffer_sampler():
            while True:
                yield self.buffer_class.uniform_sample_from_buffer(
                    rng=rng,
                    client=client,
                    buffer=buffer,
                    n_obs=self.train_n_observations_obs + self.train_n_observations_int,
                    n_int=self.train_n_observations_int,
                )

        def infinite_buffer_collate_fn(batch):
            return structured_torch_function(
                func=self.buffer_class.collate_fn, batch=batch, dtype=self.buffer_class.dtype
            )

        ds = torch.utils.data.DataLoader(
            ds,
            collate_fn=infinite_buffer_collate_fn,
            batch_size=None,
            sampler=torch.utils.data.BatchSampler(
                torch.utils.data.RandomSampler(range(1)), batch_size=self.train_n_observations_obs
            ),
            num_workers=0,
        )

    else:
        # dataset is simply the buffer
        ds = torch.utils.data.IterableDataset()

        def buffer_index_sampler():
            for i in range(buffer.max_size):
                yield i

        def buffer_index_collate_fn(batch):
            return structured_torch_function(
                func=self.buffer_class.index_into_buffer, batch=batch, client=self._main_plasma_client, buffer=buffer
            )

        ds = torch.utils.data.DataLoader(
            ds,
            collate_fn=buffer_index_collate_fn,
            batch_size=None,
            sampler=torch.utils.data.BatchSampler(
                torch.utils.data.SequentialSampler(range(buffer.max_size)), batch_size=1
            ),
            num_workers=0,
        )

    # create batches
    if batch_dims is not None:
        for i, batch_size in enumerate(reversed(batch_dims)):
            ds = torch.utils.data.DataLoader(ds, batch_size=batch_size, num_workers=0)

    ds = torch.utils.data.DataLoader(ds, prefetch_factor=self.prefetch, num_workers=0)

    if self.double_cache_train and is_train:
        ds = _double_cache(ds)

    yield from ds

def make_datasets(self, descr):
    """Generate dict of datasets {n_vars: torch.utils.data.Dataset} for config `descr`"""
    ds = {}
    is_train = descr == "train"
    for n_vars, buffer in self._state.buffers[descr].items():
        batch_dims = self.batch_dims_train[n_vars]["device"] if is_train else self.batch_dims_test[n_vars]["device"]
        ds[n_vars] = self._make_dataset(buffer, is_train=is_train, batch_dims=batch_dims)

    return ds

def make_test_datasets(self):
    test_set_ids = [k for k in self.config["data"].keys() if k != "train"]
    return {descr: self.make_datasets(descr) for descr in test_set_ids}


# data loading with this format --  