#!/usr/bin/python3
# -*- coding: utf-8 -*-


from fire import Fire
import sys
from pyhelp.utils.utils import find_object


__doc__=r"""print python documentation from command line.
Example Usage:
```bash
pyhelp.pydocs matplotlib.pyplot
pyhelp.pydocs torch.sigmoid
```
"""
    
def read_doc(object_name):
    obj = find_object(object_name)

    if obj is None:
        return f"Targets {object_name} not found"
    else:
        doc_string = obj.__doc__
        if doc_string:
            return doc_string
        else:
            return f"Target {obj.__name__} has no doc string"

def main():
    if len(sys.argv) < 2 or '-h' in sys.argv or '--help' in sys.argv:
        print(__doc__)

         
    else:
        object_name=sys.argv[1]
        string = read_doc(object_name)
        print(string)


if __name__ == '__main__':
    Fire(main)
