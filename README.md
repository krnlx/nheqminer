# Donation
If you like what im doing consider donating BTC to 1LFgMtZdnjZXVn9wnZuigiePSavgis7Ktr

# Build instructions:

## Linux + SA5
cd nheqminer/Linux_cmake/nheqminer_cuda_sa && cmake . && make

#CUDA THREADS AND BLOCKS HARDCODED 

#speed on 1070

[08:53:50][0x00007fe73631d740] Speed [300 sec]: 61.627 I/s, 115.119 Sols/s


#run


./nheqminer_cuda_sa -t 0 -cv 0 -cd 0  -cs  -l eu1-zcash.flypool.org:3333 -u t1SfBtVs3pVmGuKjXhkK6Bzb3WzGkxzU5hu.test


#It is slower then original opencl implementation, feel free to commit changes
