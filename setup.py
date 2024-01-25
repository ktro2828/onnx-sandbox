# -*- coding: utf-8 -*-
from setuptools import setup

packages = \
['onnx_sandbox']

package_data = \
{'': ['*']}

install_requires = \
['numpy>=1.26.3,<2.0.0',
 'onnx>=1.15.0,<2.0.0',
 'onnxruntime-gpu>=1.16.3,<2.0.0',
 'onnxscript>=0.1.0.dev20240125,<0.2.0',
 'opencv-python>=4.9.0.80,<5.0.0.0',
 'torch==2.1.1']

setup_kwargs = {
    'name': 'onnx-sandbox',
    'version': '0.1.0',
    'description': '',
    'long_description': '',
    'author': 'ktro2828',
    'author_email': 'kotaro.uetake@tier4.jp',
    'maintainer': 'None',
    'maintainer_email': 'None',
    'url': 'None',
    'packages': packages,
    'package_data': package_data,
    'install_requires': install_requires,
    'python_requires': '>=3.10,<4.0',
}
from build import *
build(setup_kwargs)

setup(**setup_kwargs)
