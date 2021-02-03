import importlib

def merge_name(list_of_name):
    """
        Merge ['A', 'B', 'C'] to 'A.B.C' with '.' seperation.
    """
    final_name = ""
    for name in list_of_name:
        final_name += name + "."

    final_name = final_name.strip('.')
    return final_name


def find_object(object_string:str):
    """Return the object(module, class, function) searching with string.

    Args:
        object_string (str)
    Return:
        module/class/function, None with not found

    Example:
    1. 
        import torch
        torch_module = find_object('torch')
        torch_module.sigmoid == torch.sigmoid
    2. 
        exp = find_object('numpy.exp')
        e = exp(1.0)
    """

    function_name = object_string
    splitted_names = function_name.split('.')

    is_found = False
    for i in range(len(splitted_names), -1, -1):
        try:
            merged_name = merge_name(splitted_names[0:i])
            module = importlib.import_module(merged_name)
        except:
            continue
        is_found = True
        base_obj = module
        for name in splitted_names[i:]:
            base_obj = getattr(base_obj, name)
        break
    
    if is_found:
        return base_obj
    
    else:
        return None

