export LD_LIBRARY_PATH=/workspace/zhanghandi/roformer_full_quant_envs/TensorRT-8.4.0.6/lib/:$LD_LIBRARY_PATH
PADDLE_ROOT=/workspace/zhanghandi/roformer_full_quant_envs/Paddle/
TENSORRT_ROOT=/workspace/zhanghandi/roformer_full_quant_envs/TensorRT-8.4.0.6/
#CUDNN_HOME=/workspace/zhanghandi/smooth_infer/cudnn_v8.2.4_cuda11.4/cuda/
#CUDA_LIB=/workspace/zhanghandi/smooth_infer/cuda-11.4/cuda/lib64

cd build
cmake .. -DCMAKE_BUILD_TYPE=Release \
         -DPY_VERSION=3.7 \
         -DFLUID_INFERENCE_INSTALL_DIR=$PADDLE_ROOT \
         -DWITH_PYTHON=ON \
         -DON_INFER=ON \
         -DWITH_GPU=ON \
         -DWITH_TENSORRT=ON \
         -DWITH_MKL=ON  \
         -DWITH_MKLDNN=OFF \
	 -DWITH_NCCL=OFF \
         -DTENSORRT_ROOT=${TENSORRT_ROOT} \
         #-DCUDNN_ROOT=${CUDNN_HOME} \
         #-DCUDA_TOOLKIT_ROOT_DIR=${CUDA_LIB} \
         -DCUDA_ARCH_NAME=Ampere

make -j 40
make inference_lib_dist -j4
