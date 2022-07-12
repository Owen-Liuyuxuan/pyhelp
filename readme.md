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
