mkdir build_Ampere
cd build_Ampere

cmake .. -DPY_VERSION=3.7 -DWITH_TESTING=OFF -DWITH_MKL=ON -DWITH_GPU=ON -DON_INFER=ON \
                 -DCUDA_ARCH_NAME=Ampere   -DWITH_TENSORRT=ON -DTENSORRT_ROOT=/workspace/zhanghandi/envs/TensorRT-8.4.1.5/
                 #-DCUDA_ARCH_NAME=All   -DWITH_TENSORRT=ON -DTENSORRT_ROOT=/workspace/zhanghandi/envs/TensorRT-8.4.1.5/

make -j 40
make inference_lib_dist -j4
