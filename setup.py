
from distutils.core import setup
import warnings

setup(name='lineracer',
   version='0.0.1',
   python_requires='>=3.8',
   description='A simple turn-based multiplayer racing game.',
   author='Jonas Hall',
   author_email='hall.f.jonas@gmail.com',
   license='MIT',
   include_package_data = True,
   py_modules=[],
   setup_requires=['setuptools_scm'],
   install_requires=[
      'matplotlib>=3.7.5',
      'numpy>=1.24.4,<2.0.0',
      'scipy>=1.10.1',
   ]
)

warnings.filterwarnings("always")
