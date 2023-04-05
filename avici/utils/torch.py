import torch

def structured_py_function(func, inp, Tout, name=None):
    def wrapped_func(*flat_inp):
        reconstructed_inp = torch._utils._unflatten_dense_tensors(flat_inp, inp)
        result = func(*reconstructed_inp)
        return torch._utils._flatten_dense_tensors(result)
    
    flat_Tout = torch._utils._flatten_dense_tensors(Tout)
    flat_out = torch.ops.numpy_function(
        func=wrapped_func,
        inp=torch._utils._flatten_dense_tensors(inp),
        output=(_dtype_to_tensor(v) for v in flat_Tout),
        name=name
    )
    spec_out = torch._utils._unflatten_dense_tensors(flat_Tout, Tout)
    out = torch._utils._unflatten_dense_tensors(flat_out, spec_out)
    return out

def _dtype_to_tensor(v):
    return torch.Tensor().new_empty((), dtype=v) if isinstance(v, torch.dtype) else v

def _tensor_to_dtype(v):
    return v.dtype if isinstance(v, torch.Tensor) else v
