import torch 

def torch_nanstd(tensor, dim=None, keepdim=False):
    valid_values = torch.isfinite(tensor)
    tensor_no_nan = tensor.clone()
    tensor_no_nan[~valid_values] = 0.0
    num_valid_values = valid_values.sum(dim=dim, keepdim=keepdim).float()
    
    mean = (tensor_no_nan * valid_values).sum(dim=dim, keepdim=keepdim) / num_valid_values
    var = (((tensor_no_nan - mean) ** 2) * valid_values).sum(dim=dim, keepdim=keepdim) / num_valid_values
    return torch.sqrt(var)

def torch_nanmin(tensor, dim=None, keepdim=False):
    valid_values = torch.isfinite(tensor)
    large_value = torch.finfo(tensor.dtype).max
    tensor_no_nan = tensor.clone()
    tensor_no_nan[~valid_values] = large_value
    return tensor_no_nan.min(dim=dim, keepdim=keepdim)

def _torch_cpm_standardizer(x):
    # compute library sizes (sum of row)
    x_libsize = x.sum(dim=-1, keepdim=True)

    # divide each cell by library size and multiply by 10^6
    # will yield nan for rows with zero expression and for zero expression entries
    log2cpm = torch.where(torch.isclose(x, 0.0), float('nan'), torch.log2(x / (x_libsize * 1e-6)))
    return log2cpm

def _torch_cpm_shift_scale(x, shift, scale):
    # shift and scale
    x = (x - torch.where(torch.isnan(shift), 0.0, shift)) / torch.where(torch.isnan(scale), 1.0, scale)

    # set nans (set for all-zero rows, i.e. zero libsize) to minimum (i.e. to zero)
    # catch empty arrays (when x_int is empty and has axis n=0)
    if x.nelement() != 0:
        x = torch.where(torch.isnan(x), 0.0, x)
    return x

def _torch_standardize_count(x_obs, x_int):
    # cpm normalization
    x_obs = x_obs.clone().detach()
    x_obs[..., 0] = _torch_cpm_standardizer(x_obs[..., 0])
    x_int = x_int.clone().detach()
    x_int[..., 0] = _torch_cpm_standardizer(x_int[..., 0])

    # subtract min (~robust global median) and divide by global std dev
    global_ref_x = torch.cat([x_obs[..., 0], x_int[..., 0]], dim=-2)
    global_min = torch_nanmin(global_ref_x, dim=(-1, -2), keepdim=True)
    global_std = torch_nanstd(global_ref_x, dim=(-1, -2), keepdim=True)

    x_obs = x_obs.clone().detach()
    x_obs[..., 0] = _torch_cpm_shift_scale(x_obs[..., 0], global_min, global_std)
    x_int = x_int.clone().detach()
    x_int[..., 0] = _torch_cpm_shift_scale(x_int[..., 0], global_min, global_std)

    return x_obs, x_int

def torch_standardize_count_simple(x):
    """_torch_standardize_count but with only one argument"""
    # cpm normalization
    x = x.clone().detach()
    x[..., 0] = _torch_cpm_standardizer(x[..., 0])

    # subtract min (~robust global median) and divide by global std dev
    global_ref_x = x[..., 0]
    global_min = torch_nanmin(global_ref_x, dim=(-1, -2), keepdim=True)
    global_std = torch_nanstd(global_ref_x, dim=(-1, -2), keepdim=True)

    x = x.clone().detach()
    x[..., 0] = _torch_cpm_shift_scale(x[..., 0], global_min, global_std)
    return x

def _torch_standardize_default(x_obs, x_int):
    # default z-standardization
    ref_x = torch.cat([x_obs[..., 0], x_int[..., 0]], dim=-2)
    mean = torch.mean(ref_x, dim=-2, keepdim=True)
    std = torch.std(ref_x, dim=-2, keepdim=True,unbiased=False) # 2.version requires correction=0 
    x_obs_default = x_obs.clone().detach()
    x_obs_default[..., 0] = (x_obs[..., 0] - mean) / torch.where(std == 0.0, torch.tensor(1.0, dtype=std.dtype), std)
    x_int_default = x_int.clone().detach()
    x_int_default[..., 0] = (x_int[..., 0] - mean) /torch.where(std == 0.0, torch.tensor(1.0, dtype=std.dtype), std)
    return x_obs_default, x_int_default

def torch_standardize_default_simple(x):
    """_torch_standardize_default but with only one argument"""
    # default z-standardization
    ref_x = x[..., 0]
    mean = torch.mean(ref_x, dim=-2, keepdim=True)
    std = torch.std(ref_x, dim=-2, keepdim=True,unbiased=False) # 2.version requires correction=0 
    x = x.clone().detach()
    x[..., 0] = (x[..., 0] - mean) / torch.where(std == 0.0, torch.tensor(1.0, dtype=std.dtype), std)
    return x

def torch_standardize_data(data):
    """Standardize observations `x_obs` and `x_int`"""
    x_obs = data["x_obs"]
    x_int = data["x_int"]

    assert (x_obs.shape[-1] == 1 or x_obs.shape[-1] == 2) \
       and (x_int.shape[-1] == 1 or x_int.shape[-1] == 2), \
        f"Assume concat 3D shape but got: x_obs {x_obs.shape} and x_int {x_int.shape}"

    x_obs_count, x_int_count = _torch_standardize_count(x_obs, x_int)
    x_obs_default, x_int_default = _torch_standardize_default(x_obs, x_int)
    return {
        **data,
        "x_obs": torch.where(data["is_count_data"][..., None, None, None], x_obs_count, x_obs_default),
        "x_int": torch.where(data["is_count_data"][..., None, None, None], x_int_count, x_int_default),
    }

def torch_standardize_x(x, is_count_data):
    """Standardize observations"""
    assert (x.shape[-1] == 1 or x.shape[-1] == 2),\
        f"Assume concat 3D shape but got: x {x.shape}"

    # TODO this is not a nice implementation; refactor torch_standardize_data to accept variable number of x fields in data
    dummy = torch.zeros((*x.shape[:-3], 0, *x.shape[-2:]))
    data = torch_standardize_data({"x_obs": x, "x_int": dummy, "is_count_data": is_count_data})
    return data["x_obs"]

"""Data batching"""

def torch_get_train_x(batch, p_obs_only):
    subkey1, subkey2 = torch.randperm(int(2**32-1)).split(1)
    n_obs_and_int, n_int = batch["x_obs"].shape[-3], batch["x_int"].shape[-3]

    only_observational = torch.bernoulli(torch.tensor(p_obs_only))
    x = torch.where(
        only_observational,
        # only observational data
        batch["x_obs"], # already has N=n_obs + n_int
        # mix observational and interventional
        # select `n_obs` elements of `n_obs + n_int` available observational data
        torch.cat([
            batch["x_obs"][..., torch.randperm(n_obs_and_int)[:(n_obs_and_int - n_int)], :, :],
            batch["x_int"]
        ], dim=-3),
    )
    x = x[..., torch.randperm(n_obs_and_int), :, :]
    x = torch_standardize_x(x, batch["is_count_data"])  # place 1/2 where torch data should be standardized; do not do anywhere else to avoid double-standardizing count data
    return x


def torch_get_x(batch):
    x = torch.cat([batch["x_obs"], batch["x_int"]], dim=-3)
    x = torch_standardize_x(x, batch["is_count_data"]) # place 2/2 where torch data should be standardized; do not do anywhere else to avoid double-standardizing count data
    return x
