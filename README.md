# cudss_permute 

## Build 
cuDSS path should be passed to cmake. The example below where to it is installed by default on Windows. 
```
mkdir build
cd build 
cmake -DCMAKE_PREFIX_PATH="C:\Program Files\NVIDIA cuDSS\v0.6\lib\12\cmake" ..
```

Depending on the system, this will generate either a `.sln` project on Windows or a `make` file for a Linux system. 

## Run 
```
./test_cudss matrix.mtx <permute_type>
```
The `<permute_type>` argument can be one of the following:

- `default`
- `metis`
- `symrcm`
- `symamd`

A few example `.mtx` matrices are provided in the `/data` directory. A larger matrix is also provided as a `.zip` file; you can decompress it for more intensive testing.