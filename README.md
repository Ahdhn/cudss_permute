# cudss_permute [![Windows](https://github.com/Ahdhn/cudss_permute/actions/workflows/Windows.yml/badge.svg)](https://github.com/Ahdhn/cudss_permute/actions/workflows/Windows.yml) [![Ubuntu](https://github.com/Ahdhn/cudss_permute/actions/workflows/Ubuntu.yml/badge.svg)](https://github.com/Ahdhn/cudss_permute/actions/workflows/Ubuntu.yml)


## Build 
cuDSS path should be passed to cmake. The example below where to it is installed by default on Windows. 
```
mkdir build
cd build 
cmake -DCMAKE_PREFIX_PATH="C:\Program Files\NVIDIA cuDSS\v0.6\lib\12\cmake" ..
```

Depending on the system, this will generate either a `.sln` project on Windows or a `make` file for a Linux system. 
