from setuptools import setup, find_packages

setup(name='DataQuality',
      version='1.0',
      description='Library to assess data quality',
      # This is not the complete set of packages, but should be enough to
      # do a local install of the package in editable mode, via `pip install -e .`
      install_requires=[
          "numpy",
          "torch",
          "matplotlib",
          "scipy",
          "torchvision",
          "scikit-image",
          "h5py",
          "pandas",
      ],
      packages=find_packages(exclude=('tests', 'docs', 'build'))
      )
