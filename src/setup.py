from setuptools import *

LONG_DESC = """
A bunch of utilities for python, jupyter notebook and so on.
"""

install_requires=[
    'numpy', 'ipython', 'matplotlib', 'fire'
]

setup(name='pyhelp',
	  version='0.1.0',
	  description='Utilities for python',
	  long_description=LONG_DESC,
	  author='yuxuan',
	  install_requires=install_requires,
	  author_email='yliuhb@connect.ust.hk',
	  license='MIT',
	  packages=find_packages(),
	  entry_points={
        "console_scripts": [
            "pyhelp.pydocs=pyhelp.cli.read_docs:main",
			"pyhelp.kitti2coco=pyhelp.cli.kitti2coco:main",
			"pyhelp.kitti2custom=pyhelp.cli.kitti2custom:main",
			"pyhelp.mmdet2kitti=pyhelp.cli.mmdet2kitti:main",
			"pyhelp=pyhelp.cli.introduction:main"
        ],
    },
	  zip_safe=False)