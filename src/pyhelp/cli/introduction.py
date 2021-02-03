import sys
from pyhelp.cli.read_docs import read_doc
def main():
    command_line_dict={
        "pyhelp.pydocs": "pyhelp.cli.read_docs",
		"pyhelp.kitti2coco" : "pyhelp.cli.kitti2coco",
		"pyhelp.kitti2custom" : "pyhelp.cli.kitti2custom",
		"pyhelp.mmdet2kitti" : "pyhelp.cli.mmdet2kitti"
    }
    if len(sys.argv) < 2 or '-h' in sys.argv or '--help' in sys.argv:
        print("Watch command line helping by typing: 'pyhelp <key>' in command line \n")
        print("Available keys:")
        print(list(command_line_dict.keys()))
        exit()
    else:
        name = sys.argv[1]
        print(f"Help for {name}:")
        print(read_doc(command_line_dict[name]))
