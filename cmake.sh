mkdir build
cd build

cmake .. -DPY_VERSION=3.7 -DWITH_TESTING=OFF -DWITH_MKL=ON -DWITH_GPU=ON -DON_INFER=ON \
                 -DWITH_TENSORRT=ON -DTENSORRT_ROOT=/workspace/zhanghandi/dx_fix_bug/envs/TensorRT-8.4.0.6/
                 #-DCUDA_ARCH_NAME=All   -DWITH_TENSORRT=ON -DTENSORRT_ROOT=/workspace/zhanghandi/envs/TensorRT-8.4.1.5/

make -j 40
make inference_lib_dist -j4
