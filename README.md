CS5242-neural-motion-planning

# Environment
Tested with:<br>
**Python 3.8**<br>
**Ubuntu18.04**

# Installation instructions:

## Install dependencies of OMPL
https://github.com/ompl/ompl/blob/main/doc/markdown/installPyPlusPlus.md

## Install OMPL from source
It is very important that you compile ompl with the correct python version with the CMake flag.
```
git clone https://github.com/ompl/ompl.git
mkdir build/Release
cd build/Release
cmake ../.. -DPYTHON_EXEC=/path/to/python-X.Y # This is important!!! Make sure you are pointing to the correct python version.
make -j 4 update_bindings # replace "4" with the number of cores on your machine. This step takes some time.
make -j 4 # replace "4" with the number of cores on your machine
```

## Install Pybullet
Just install pybullet normally.
```
pip install pybullet
```

# Other packages:
pytorch
networkx
matplotlib
numpy