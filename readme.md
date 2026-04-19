# Pyhelp

**tons of w.t.f. scripts and helpers**

## Installation

Generally it requires pytorch, numpy, matplotlib, jupyter and so on.
```bash
./install.sh
```

## Command Line tools

Command line run
```bash
pyhelp
```
to get the supported command line tools.

For each tools like pyhelp.pydocs run
```bash
pyhelp.pydocs --help
```
to get the usage.

## Debugging Tools

The code contain code for debugging pytorch related code in vscode. Espescially using the ["debug console"](https://code.visualstudio.com/docs/editor/debugging#_debug-console-repl). 

- Visualizing image/feature/result tensors/numpy arrays.
- Save and load numpy arrays / torch tensors to disk for inspection or later reuse.
- Estimate the time of a certain function call.
- Detailling profile a certain function call.


### Visualizing

When debug the pytorch code, and want to visualize the image/feature/result tensors or numpy array without changing the original code.

At the debug console, run

```python
from pyhelp.debug_utils import imshow
visualized_image = torch.zeros([B, C, H, W]) # even required gradients/in gpu is ok.
imshow(visualized_image, **matplotlibarguments)
```

How it works:
- Detach/transfer any tensor to numpy, or stay in numpy
- Extract the first tensors if fed with a batch of tensors.
- Infer from the shape, whether the visualized array is a heatmap / an rgb image / a feature map. And whether we need to tranpose from $3HW$ to $HW3$. The logic is hard-coded (not difficult to read) and works at most debugging cases.
- The code will both show the figure with matplotlib imshow;show (it will try to be interactive); and will also save the figure at debug.png
- If the input is a **list** of image/tensor, it will try to organized the input into a grid using the pyplot.subplot api. Other performances are the same.

### Data I/O

Save tensors or arrays from the debug console (or anywhere) into a single `.npz` file: the array data is stored with JSON metadata so you can restore the original type (numpy vs torch), dtype, shape, and for tensors the original device and `requires_grad`. Bfloat16 tensors are saved as float32 in the payload; loading can cast back.

```python
from pyhelp.debug_utils import save_data, load_data
from pyhelp.debug_utils.data_io import peek_data_metadata
# save_data / load_data accept numpy.ndarray or torch.Tensor
path = save_data(my_tensor)  # path/name optional; default stem inferred from call-site when possible
obj = load_data(path)  # restores numpy or torch like it was saved

# Force output kind or destination
arr = load_data(path, as_type="numpy")
t = load_data(path, as_type="torch", device="cuda")

meta = peek_data_metadata(path)  # read meta only without loading the full array
```

Typed helpers (`save_array` / `load_array`, `save_tensor` / `load_tensor`) restrict inputs and return types. Use `save_data(..., path=..., name=..., compressed=True, overwrite=True)` when you do not want automatic naming.

### Timing or Profiling

```python
from pyhelp.debug_utils import timer, profiler
# Just a timer
original_result = timer(function, *func_args, **func_kwargs) # the code will print out the time spent.

# Detail timer
result_dict = profiler(function, *func_args, **func_kwargs)
result_dict['result'] # original_result
result_dict['prof'] #profile result from torch.autograd.profiler.profile
```
