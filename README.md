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

A few example `.mtx` matrices are provided in the `/data` directory. Several larger matrices are also provided as `.zip` files. You can decompress it for more intensive testing.

## Results 
Using machine with the following specs:
- Windows Visual Studio 2022
- CUDA 12.6
- cuDSS 0.6 
- RTX 4090

All timing are in ms. Note that `w/ metis` permutation column does *not* include the time that METIS took to the permutation. This is the permutation time from cuDSS. 

| Matrix | N | NNZ | w/ metis |   |   |   | w/ default |   |   |   |
|:------|:--:|:---:|:--------:|:--:|:--:|:--:|:---------:|:--:|:--:|:--:|
|        |    |     | Permutation | Symbolic | Factorize | Solve | Permutation | Symbolic | Factorize | Solve |
| `loop_A.mtx` | 173440  | 1214560 | 10.76 | 314.3 | 50.6 | 14.2 | 496.4 | 25.2 | 14.3 | 4.48| 
| `edgar-allan-poe-1_A.mtx` | 194632  | 1362412 | 11.0 | 376.2 |51.5 | 16.3| 605.3 | 27.68 | 14.9 | 5.7 |
| `Sphere8_A.mtx` | 393218  | 2752514 | 16.7 | 738.3 |135.8 | 32.9| 1220.6 | 63.7 | 41.2 | 6.6 |
| ` Nefertiti_A.mtx` | 1009118  | 7063814  | 34.4| 2051.8 | 328.6 | 89.1 | 3613.4 | 114.98 | 71.9 | 25.2 |


