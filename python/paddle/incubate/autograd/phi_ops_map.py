op_map = {
    "abs": {
        "phi_name": "abs",
        "inputs": {
            "x": "X"
        }
    },
    "acos": {
        "phi_name": "acos",
        "inputs": {
            "x": "X"
        }
    },
    "acosh": {
        "phi_name": "acosh",
        "inputs": {
            "x": "X"
        }
    },
    "elementwise_add": {
        "phi_name": "add"
    },
    "addmm": {
        "phi_name": "addmm",
        "inputs": {
            "input": "Input",
            "x": "X",
            "y": "Y"
        },
        "attrs": {
            "alpha": "Alpha",
            "beta": "Beta"
        }
    },
    "affine_grid": {
        "phi_name": "affine_grid"
    },
    "allclose": {
        "phi_name": "allclose",
        "inputs": {
            "x": "Input",
            "y": "Other"
        },
        "scalar": {
            "rtol": {
                "data_type": "std::string",
                "tensor_name": "Rtol"
            },
            "atol": {
                "data_type": "std::string",
                "tensor_name": "Atol"
            }
        }
    },
    "angle": {
        "phi_name": "angle",
        "inputs": {
            "x": "X"
        }
    },
    "argsort": {
        "phi_name": "argsort",
        "inputs": {
            "x": "X"
        }
    },
    "as_complex": {
        "phi_name": "as_complex",
        "inputs": {
            "x": "X"
        }
    },
    "as_real": {
        "phi_name": "as_real",
        "inputs": {
            "x": "X"
        }
    },
    "asin": {
        "phi_name": "asin",
        "inputs": {
            "x": "X"
        }
    },
    "asinh": {
        "phi_name": "asinh",
        "inputs": {
            "x": "X"
        }
    },
    "atan": {
        "phi_name": "atan",
        "inputs": {
            "x": "X"
        }
    },
    "atan2": {
        "phi_name": "atan2",
        "inputs": {
            "x": "X1",
            "y": "X2"
        }
    },
    "atanh": {
        "phi_name": "atanh",
        "inputs": {
            "x": "X"
        }
    },
    "batch_norm": {
        "phi_name": "batch_norm"
    },
    "bernoulli": {
        "phi_name": "bernoulli",
        "inputs": {
            "x": "X"
        }
    },
    "bicubic_interp_v2": {
        "phi_name": "bicubic_interp"
    },
    "bilinear_interp_v2": {
        "phi_name": "bilinear_interp"
    },
    "bmm": {
        "phi_name": "bmm",
        "inputs": {
            "x": "X",
            "y": "Y"
        }
    },
    "ceil": {
        "phi_name": "ceil",
        "inputs": {
            "x": "X"
        }
    },
    "celu": {
        "phi_name": "celu",
        "inputs": {
            "x": "X"
        }
    },
    "cholesky": {
        "phi_name": "cholesky",
        "inputs": {
            "x": "X"
        }
    },
    "cholesky_solve": {
        "phi_name": "cholesky_solve",
        "inputs": {
            "x": "X",
            "y": "Y"
        }
    },
    "clip": {
        "phi_name": "clip",
        "inputs": {
            "x": "X"
        },
        "scalar": {
            "min": {
                "data_type": "float",
                "tensor_name": "Min"
            },
            "max": {
                "data_type": "float",
                "tensor_name": "Max"
            }
        }
    },
    "complex": {
        "phi_name": "complex",
        "inputs": {
            "real": "X",
            "imag": "Y"
        }
    },
    "concat": {
        "phi_name": "concat"
    },
    "conditional_block": {
        "phi_name": "conditional_block"
    },
    "conj": {
        "phi_name": "conj",
        "inputs": {
            "x": "X"
        }
    },
    "conv2d": {
        "phi_name": "conv2d"
    },
    "conv2d_fusion": {
        "phi_name": "conv2d_fusion"
    },
    "conv2d_transpose": {
        "phi_name": "conv2d_transpose"
    },
    "conv3d": {
        "phi_name": "conv3d"
    },
    "conv3d_transpose": {
        "phi_name": "conv3d_transpose"
    },
    "cos": {
        "phi_name": "cos",
        "inputs": {
            "x": "X"
        }
    },
    "cosh": {
        "phi_name": "cosh",
        "inputs": {
            "x": "X"
        }
    },
    "crop_tensor": {
        "phi_name": "crop",
        "inputs": {
            "x": "X"
        },
        "int_array": {
            "shape": {
                "data_type": "int",
                "tensor_name": "Shape",
                "tensors_name": "ShapeTensor"
            },
            "offsets": {
                "data_type": "int",
                "tensor_name": "Offsets",
                "tensors_name": "OffsetsTensor"
            }
        }
    },
    "cross": {
        "phi_name": "cross",
        "inputs": {
            "x": "X",
            "y": "Y"
        },
        "attrs": {
            "axis": "dim"
        }
    },
    "data_norm": {
        "phi_name": "data_norm"
    },
    "depthwise_conv2d": {
        "phi_name": "depthwise_conv2d"
    },
    "depthwise_conv2d_transpose": {
        "phi_name": "depthwise_conv2d_transpose"
    },
    "dequantize_linear": {
        "phi_name": "dequantize_linear"
    },
    "determinant": {
        "phi_name": "det",
        "inputs": {
            "x": "Input"
        }
    },
    "diag_v2": {
        "phi_name": "diag",
        "inputs": {
            "x": "X"
        }
    },
    "diag_embed": {
        "phi_name": "diag_embed",
        "inputs": {
            "input": "Input"
        }
    },
    "diagonal": {
        "phi_name": "diagonal",
        "inputs": {
            "x": "Input"
        }
    },
    "digamma": {
        "phi_name": "digamma",
        "inputs": {
            "x": "X"
        }
    },
    "dist": {
        "phi_name": "dist",
        "inputs": {
            "x": "X",
            "y": "Y"
        }
    },
    "distributed_push_sparse": {
        "phi_name": "distributed_push_sparse"
    },
    "elementwise_div": {
        "phi_name": "divide"
    },
    "dot": {
        "phi_name": "dot",
        "inputs": {
            "x": "X",
            "y": "Y"
        }
    },
    "dropout": {
        "phi_name": "dropout"
    },
    "dropout_nd": {
        "phi_name": "dropout_nd"
    },
    "eig": {
        "phi_name": "eig",
        "inputs": {
            "x": "X"
        }
    },
    "eigh": {
        "phi_name": "eigh",
        "inputs": {
            "x": "X"
        }
    },
    "eigvals": {
        "phi_name": "eigvals",
        "inputs": {
            "x": "X"
        }
    },
    "elementwise_pow": {
        "phi_name": "elementwise_pow"
    },
    "elu": {
        "phi_name": "elu",
        "inputs": {
            "x": "X"
        }
    },
    "lookup_table_v2": {
        "phi_name": "embedding"
    },
    "equal_all": {
        "phi_name": "equal_all",
        "inputs": {
            "x": "X",
            "y": "Y"
        }
    },
    "erf": {
        "phi_name": "erf",
        "inputs": {
            "x": "X"
        }
    },
    "erfinv": {
        "phi_name": "erfinv",
        "inputs": {
            "x": "X"
        }
    },
    "exp": {
        "phi_name": "exp",
        "inputs": {
            "x": "X"
        }
    },
    "expand_v2": {
        "phi_name": "expand"
    },
    "expm1": {
        "phi_name": "expm1",
        "inputs": {
            "x": "X"
        }
    },
    "fake_channel_wise_quantize_abs_max": {
        "phi_name": "fake_channel_wise_quantize_abs_max"
    },
    "fake_channel_wise_quantize_dequantize_abs_max": {
        "phi_name": "fake_channel_wise_quantize_dequantize_abs_max"
    },
    "fake_quantize_abs_max": {
        "phi_name": "fake_quantize_abs_max"
    },
    "fake_quantize_dequantize_abs_max": {
        "phi_name": "fake_quantize_dequantize_abs_max"
    },
    "fake_quantize_dequantize_moving_average_abs_max": {
        "phi_name": "fake_quantize_dequantize_moving_average_abs_max"
    },
    "fake_quantize_moving_average_abs_max": {
        "phi_name": "fake_quantize_moving_average_abs_max"
    },
    "fake_quantize_range_abs_max": {
        "phi_name": "fake_quantize_range_abs_max"
    },
    "fft_c2c": {
        "phi_name": "fft_c2c",
        "inputs": {
            "x": "X"
        }
    },
    "fft_c2r": {
        "phi_name": "fft_c2r",
        "inputs": {
            "x": "X"
        }
    },
    "fft_r2c": {
        "phi_name": "fft_r2c",
        "inputs": {
            "x": "X"
        }
    },
    "fill_diagonal": {
        "phi_name": "fill_diagonal",
        "inputs": {
            "x": "X"
        }
    },
    "fill_diagonal_tensor": {
        "phi_name": "fill_diagonal_tensor",
        "inputs": {
            "x": "X",
            "y": "Y"
        }
    },
    "flip": {
        "phi_name": "flip",
        "inputs": {
            "x": "X"
        }
    },
    "floor": {
        "phi_name": "floor",
        "inputs": {
            "x": "X"
        }
    },
    "elementwise_floordiv": {
        "phi_name": "floor_divide"
    },
    "elementwise_fmax": {
        "phi_name": "fmax"
    },
    "elementwise_fmin": {
        "phi_name": "fmin"
    },
    "fold": {
        "phi_name": "fold",
        "inputs": {
            "x": "X"
        }
    },
    "frame": {
        "phi_name": "frame",
        "inputs": {
            "x": "X"
        }
    },
    "frobenius_norm": {
        "phi_name": "frobenius_norm"
    },
    "fill_constant": {
        "phi_name": "full"
    },
    "gather": {
        "phi_name": "gather"
    },
    "gather_nd": {
        "phi_name": "gather_nd",
        "inputs": {
            "x": "X",
            "index": "Index"
        }
    },
    "gather_tree": {
        "phi_name": "gather_tree",
        "inputs": {
            "ids": "Ids",
            "parents": "Parents"
        }
    },
    "gelu": {
        "phi_name": "gelu",
        "inputs": {
            "x": "X"
        }
    },
    "grad_add": {
        "phi_name": "grad_add"
    },
    "grid_sampler": {
        "phi_name": "grid_sample",
        "inputs": {
            "x": "X",
            "grid": "Grid"
        }
    },
    "gru": {
        "phi_name": "gru"
    },
    "gumbel_softmax": {
        "phi_name": "gumbel_softmax",
        "inputs": {
            "x": "X"
        }
    },
    "hard_swish": {
        "phi_name": "hard_swish"
    },
    "hard_shrink": {
        "phi_name": "hardshrink",
        "inputs": {
            "x": "X"
        }
    },
    "hard_sigmoid": {
        "phi_name": "hardsigmoid",
        "inputs": {
            "x": "X"
        }
    },
    "brelu": {
        "phi_name": "hardtanh",
        "inputs": {
            "x": "X"
        }
    },
    "elementwise_heaviside": {
        "phi_name": "heaviside"
    },
    "histogram": {
        "phi_name": "histogram",
        "inputs": {
            "input": "X"
        }
    },
    "imag": {
        "phi_name": "imag",
        "inputs": {
            "x": "X"
        }
    },
    "index_sample": {
        "phi_name": "index_sample",
        "inputs": {
            "x": "X",
            "index": "Index"
        }
    },
    "index_select": {
        "phi_name": "index_select",
        "inputs": {
            "x": "X",
            "index": "Index"
        },
        "attrs": {
            "axis": "dim"
        }
    },
    "inplace_abn": {
        "phi_name": "inplace_abn"
    },
    "inverse": {
        "phi_name": "inverse",
        "inputs": {
            "x": "Input"
        }
    },
    "is_empty": {
        "phi_name": "is_empty",
        "inputs": {
            "x": "X"
        }
    },
    "isclose": {
        "phi_name": "isclose",
        "inputs": {
            "x": "Input",
            "y": "Other"
        },
        "scalar": {
            "rtol": {
                "data_type": "std::string",
                "tensor_name": "Rtol"
            },
            "atol": {
                "data_type": "std::string",
                "tensor_name": "Atol"
            }
        }
    },
    "isfinite_v2": {
        "phi_name": "isfinite",
        "inputs": {
            "x": "X"
        }
    },
    "isinf_v2": {
        "phi_name": "isinf",
        "inputs": {
            "x": "X"
        }
    },
    "isnan_v2": {
        "phi_name": "isnan",
        "inputs": {
            "x": "X"
        }
    },
    "kthvalue": {
        "phi_name": "kthvalue",
        "inputs": {
            "x": "X"
        }
    },
    "label_smooth": {
        "phi_name": "label_smooth",
        "inputs": {
            "label": "X",
            "prior_dist": "PriorDist"
        }
    },
    "layer_norm": {
        "phi_name": "layer_norm"
    },
    "leaky_relu": {
        "phi_name": "leaky_relu",
        "inputs": {
            "x": "X"
        },
        "attrs": {
            "negative_slope": "alpha"
        }
    },
    "lerp": {
        "phi_name": "lerp",
        "inputs": {
            "x": "X",
            "y": "Y",
            "weight": "Weight"
        }
    },
    "lgamma": {
        "phi_name": "lgamma",
        "inputs": {
            "x": "X"
        }
    },
    "linear_interp_v2": {
        "phi_name": "linear_interp"
    },
    "log": {
        "phi_name": "log",
        "inputs": {
            "x": "X"
        }
    },
    "log10": {
        "phi_name": "log10",
        "inputs": {
            "x": "X"
        }
    },
    "log1p": {
        "phi_name": "log1p",
        "inputs": {
            "x": "X"
        }
    },
    "log2": {
        "phi_name": "log2",
        "inputs": {
            "x": "X"
        }
    },
    "log_loss": {
        "phi_name": "log_loss",
        "inputs": {
            "input": "Predicted",
            "label": "Labels"
        }
    },
    "log_softmax": {
        "phi_name": "log_softmax"
    },
    "logit": {
        "phi_name": "logit",
        "inputs": {
            "x": "X"
        }
    },
    "logsigmoid": {
        "phi_name": "logsigmoid",
        "inputs": {
            "x": "X"
        }
    },
    "lrn": {
        "phi_name": "lrn"
    },
    "lu_unpack": {
        "phi_name": "lu_unpack",
        "inputs": {
            "x": "X",
            "y": "Pivots"
        }
    },
    "masked_select": {
        "phi_name": "masked_select",
        "inputs": {
            "x": "X",
            "mask": "Mask"
        }
    },
    "matmul_v2": {
        "phi_name": "matmul"
    },
    "mul": {
        "phi_name": "matmul_with_flatten"
    },
    "matrix_power": {
        "phi_name": "matrix_power",
        "inputs": {
            "x": "X"
        }
    },
    "elementwise_max": {
        "phi_name": "maximum"
    },
    "elementwise_min": {
        "phi_name": "maximum"
    },
    "maxout": {
        "phi_name": "maxout",
        "inputs": {
            "x": "X"
        }
    },
    "meshgrid": {
        "phi_name": "meshgrid",
        "inputs": {
            "inputs": "X"
        }
    },
    "mish": {
        "phi_name": "mish"
    },
    "mode": {
        "phi_name": "mode",
        "inputs": {
            "x": "X"
        }
    },
    "multi_dot": {
        "phi_name": "multi_dot",
        "inputs": {
            "x": "X"
        }
    },
    "multinomial": {
        "phi_name": "multinomial",
        "inputs": {
            "x": "X"
        },
        "scalar": {
            "num_samples": {
                "data_type": "int",
                "support_tensor": "True"
            }
        }
    },
    "multiplex": {
        "phi_name": "multiplex",
        "inputs": {
            "inputs": "X",
            "index": "Ids"
        }
    },
    "elementwise_mul": {
        "phi_name": "multiply"
    },
    "mv": {
        "phi_name": "mv",
        "inputs": {
            "x": "X",
            "vec": "Vec"
        }
    },
    "nce": {
        "phi_name": "nce"
    },
    "nearest_interp_v2": {
        "phi_name": "nearest_interp"
    },
    "nll_loss": {
        "phi_name": "nll_loss",
        "inputs": {
            "input": "X",
            "label": "Label",
            "weight": "Weight"
        }
    },
    "size": {
        "phi_name": "numel",
        "inputs": {
            "x": "Input"
        }
    },
    "overlap_add": {
        "phi_name": "overlap_add",
        "inputs": {
            "x": "X"
        }
    },
    "pad2d": {
        "phi_name": "pad2d"
    },
    "pad3d": {
        "phi_name": "pad3d"
    },
    "partial_sum": {
        "phi_name": "partial_sum"
    },
    "pixel_shuffle": {
        "phi_name": "pixel_shuffle",
        "inputs": {
            "x": "X"
        }
    },
    "poisson": {
        "phi_name": "poisson",
        "inputs": {
            "x": "X"
        }
    },
    "pool2d": {
        "phi_name": "pool2d"
    },
    "pool3d": {
        "phi_name": "pool3d"
    },
    "prelu": {
        "phi_name": "prelu"
    },
    "put_along_axis": {
        "phi_name": "put_along_axis",
        "inputs": {
            "arr": "Input",
            "indices": "Index",
            "values": "Value"
        },
        "attrs": {
            "axis": "Axis",
            "reduce": "Reduce"
        }
    },
    "qr": {
        "phi_name": "qr",
        "inputs": {
            "x": "X"
        }
    },
    "quantize_linear": {
        "phi_name": "quantize_linear"
    },
    "real": {
        "phi_name": "real",
        "inputs": {
            "x": "X"
        }
    },
    "reciprocal": {
        "phi_name": "reciprocal",
        "inputs": {
            "x": "X"
        }
    },
    "reduce_all": {
        "phi_name": "reduce_all"
    },
    "reduce_amax": {
        "phi_name": "reduce_amax"
    },
    "reduce_amin": {
        "phi_name": "reduce_amin"
    },
    "reduce_any": {
        "phi_name": "reduce_any"
    },
    "reduce_max": {
        "phi_name": "reduce_max"
    },
    "reduce_mean": {
        "phi_name": "reduce_mean"
    },
    "reduce_min": {
        "phi_name": "reduce_min"
    },
    "reduce_prod": {
        "phi_name": "reduce_prod"
    },
    "reduce_sum": {
        "phi_name": "reduce_sum"
    },
    "relu": {
        "phi_name": "relu",
        "inputs": {
            "x": "X"
        }
    },
    "relu6": {
        "phi_name": "relu6"
    },
    "elementwise_mod": {
        "phi_name": "remainder"
    },
    "renorm": {
        "phi_name": "renorm",
        "inputs": {
            "x": "X"
        }
    },
    "roll": {
        "phi_name": "roll",
        "inputs": {
            "x": "X"
        },
        "int_array": {
            "shifts": {
                "data_type": "int64_t",
                "tensor_name": "ShiftsTensor"
            }
        }
    },
    "round": {
        "phi_name": "round",
        "inputs": {
            "x": "X"
        }
    },
    "rsqrt": {
        "phi_name": "rsqrt",
        "inputs": {
            "x": "X"
        }
    },
    "scale": {
        "phi_name": "scale"
    },
    "scatter": {
        "phi_name": "scatter",
        "inputs": {
            "x": "X",
            "index": "Ids",
            "updates": "Updates"
        }
    },
    "scatter_nd_add": {
        "phi_name": "scatter_nd_add",
        "inputs": {
            "x": "X",
            "index": "Index",
            "updates": "Updates"
        }
    },
    "searchsorted": {
        "phi_name": "searchsorted",
        "inputs": {
            "sorted_sequence": "SortedSequence",
            "values": "Values"
        }
    },
    "seed": {
        "phi_name": "seed"
    },
    "selu": {
        "phi_name": "selu",
        "inputs": {
            "x": "X"
        }
    },
    "graph_send_recv": {
        "phi_name": "send_u_recv",
        "inputs": {
            "x": "X",
            "src_index": "Src_index",
            "dst_index": "Dst_index"
        },
        "int_array": {
            "out_size": {
                "data_type": "int64_t",
                "tensor_name": "Out_size"
            }
        }
    },
    "graph_send_ue_recv": {
        "phi_name": "send_ue_recv",
        "inputs": {
            "x": "X",
            "y": "Y",
            "src_index": "Src_index",
            "dst_index": "Dst_index"
        },
        "int_array": {
            "out_size": {
                "data_type": "int64_t",
                "tensor_name": "Out_size"
            }
        }
    },
    "graph_send_uv": {
        "phi_name": "send_uv"
    },
    "sequence_softmax": {
        "phi_name": "sequence_softmax"
    },
    "shape": {
        "phi_name": "shape"
    },
    "shard_index": {
        "phi_name": "shard_index",
        "inputs": {
            "input": "X"
        }
    },
    "share_buffer": {
        "phi_name": "share_buffer",
        "inputs": {
            "x": "X"
        }
    },
    "shuffle_channel": {
        "phi_name": "shuffle_channel"
    },
    "sigmoid": {
        "phi_name": "sigmoid",
        "inputs": {
            "x": "X"
        }
    },
    "silu": {
        "phi_name": "silu",
        "inputs": {
            "x": "X"
        }
    },
    "sin": {
        "phi_name": "sin",
        "inputs": {
            "x": "X"
        }
    },
    "sinh": {
        "phi_name": "sinh",
        "inputs": {
            "x": "X"
        }
    },
    "slice": {
        "phi_name": "slice"
    },
    "slogdeterminant": {
        "phi_name": "slogdet",
        "inputs": {
            "x": "Input"
        }
    },
    "softmax": {
        "phi_name": "softmax",
        "inputs": {
            "x": "X"
        }
    },
    "softplus": {
        "phi_name": "softplus",
        "inputs": {
            "x": "X"
        }
    },
    "softshrink": {
        "phi_name": "softshrink",
        "inputs": {
            "x": "X"
        },
        "attrs": {
            "threshold": "lambda"
        }
    },
    "softsign": {
        "phi_name": "softsign",
        "inputs": {
            "x": "X"
        }
    },
    "solve": {
        "phi_name": "solve",
        "inputs": {
            "x": "X",
            "y": "Y"
        }
    },
    "sqrt": {
        "phi_name": "sqrt",
        "inputs": {
            "x": "X"
        }
    },
    "square": {
        "phi_name": "square",
        "inputs": {
            "x": "X"
        }
    },
    "squeeze2": {
        "phi_name": "squeeze",
        "inputs": {
            "x": "X"
        },
        "attrs": {
            "axis": "axes"
        },
        "int_array": {
            "axis": {
                "data_type": "int",
                "support_tensor": "True"
            }
        }
    },
    "stack": {
        "phi_name": "stack",
        "inputs": {
            "x": "X"
        }
    },
    "elementwise_sub": {
        "phi_name": "subtract"
    },
    "svd": {
        "phi_name": "svd",
        "inputs": {
            "x": "X"
        }
    },
    "swish": {
        "phi_name": "swish"
    },
    "sync_batch_norm": {
        "phi_name": "sync_batch_norm"
    },
    "take_along_axis": {
        "phi_name": "take_along_axis",
        "inputs": {
            "arr": "Input",
            "indices": "Index"
        },
        "attrs": {
            "axis": "Axis"
        }
    },
    "tan": {
        "phi_name": "tan",
        "inputs": {
            "x": "X"
        }
    },
    "tanh": {
        "phi_name": "tanh",
        "inputs": {
            "x": "X"
        }
    },
    "tanh_shrink": {
        "phi_name": "tanh_shrink",
        "inputs": {
            "x": "X"
        }
    },
    "thresholded_relu": {
        "phi_name": "thresholded_relu",
        "inputs": {
            "x": "X"
        }
    },
    "tile": {
        "phi_name": "tile",
        "inputs": {
            "x": "X"
        },
        "int_array": {
            "repeat_times": {
                "data_type": "int",
                "tensor_name": "RepeatTimes",
                "tensors_name": "repeat_times_tensor"
            }
        }
    },
    "top_k_v2": {
        "phi_name": "topk",
        "inputs": {
            "x": "X"
        },
        "scalar": {
            "k": {
                "data_type": "int",
                "tensor_name": "K"
            }
        }
    },
    "trace": {
        "phi_name": "trace",
        "inputs": {
            "x": "Input"
        }
    },
    "transpose2": {
        "phi_name": "transpose"
    },
    "trilinear_interp_v2": {
        "phi_name": "trilinear_interp"
    },
    "trunc": {
        "phi_name": "trunc",
        "inputs": {
            "input": "X"
        }
    },
    "unbind": {
        "phi_name": "unbind",
        "inputs": {
            "input": "X"
        }
    },
    "unfold": {
        "phi_name": "unfold",
        "inputs": {
            "x": "X"
        }
    },
    "unique_consecutive": {
        "phi_name": "unique_consecutive",
        "inputs": {
            "x": "X"
        }
    },
    "unsqueeze2": {
        "phi_name": "unsqueeze",
        "inputs": {
            "x": "X"
        },
        "attrs": {
            "axis": "axes"
        },
        "int_array": {
            "axis": {
                "data_type": "int",
                "tensor_name": "AxesTensor",
                "tensors_name": "AxesTensorList"
            }
        }
    },
    "unstack": {
        "phi_name": "unstack",
        "inputs": {
            "x": "X"
        }
    },
    "viterbi_decode": {
        "phi_name": "viterbi_decode",
        "inputs": {
            "potentials": "Input",
            "transition_params": "Transition",
            "lengths": "Length"
        }
    },
    "where": {
        "phi_name": "where",
        "inputs": {
            "condition": "Condition",
            "x": "X",
            "y": "Y"
        }
    },
    "while": {
        "phi_name": "while"
    }
}
op_info = {
    "abs": {
        "args": "Tensor x",
        "output": "Tensor"
    },
    "accuracy": {
        "args": "Tensor x, Tensor indices, Tensor label",
        "output": "Tensor(accuracy), Tensor(correct), Tensor(total)"
    },
    "adadelta_": {
        "args": "Tensor param, Tensor grad, Tensor avg_squared_grad, Tensor avg_squared_update, float rho, float epsilon",
        "output": "Tensor(param_out), Tensor(moment_out), Tensor(inf_norm_out)"
    },
    "adagrad_": {
        "args": "Tensor param, Tensor grad, Tensor moment, Tensor learning_rate, float epsilon",
        "output": "Tensor(param_out), Tensor(moment_out)"
    },
    "adam_": {
        "args": "Tensor param, Tensor grad, Tensor learning_rate, Tensor moment1, Tensor moment2, Tensor beta1_pow, Tensor beta2_pow, Tensor master_param, Tensor skip_update, Scalar beta1, Scalar beta2, Scalar epsilon, bool lazy_mode, int64_t min_row_size_to_use_multithread, bool multi_precision, bool use_global_beta_pow",
        "output": "Tensor(param_out), Tensor(moment1_out), Tensor(moment2_out), Tensor(beta1_pow_out), Tensor(beta2_pow_out), Tensor(master_param_outs)"
    },
    "adamax_": {
        "args": "Tensor param, Tensor grad, Tensor learning_rate, Tensor moment, Tensor inf_norm, Tensor beta1_pow, float beta1, float beta2, float epsilon",
        "output": "Tensor(param_out), Tensor(avg_squared_grad_out), Tensor(avg_squared_update_out)"
    },
    "adamw_": {
        "args": "Tensor param, Tensor grad, Tensor learning_rate, Tensor moment1, Tensor moment2, Tensor beta1_pow, Tensor beta2_pow, Tensor master_param, Tensor skip_update, Scalar beta1, Scalar beta2, Scalar epsilon, float lr_ratio, float coeff, bool with_decay, bool lazy_mode, int64_t min_row_size_to_use_multithread, bool multi_precision, bool use_global_beta_pow",
        "output": "Tensor(param_out), Tensor(moment1_out), Tensor(moment2_out), Tensor(beta1_pow_out), Tensor(beta2_pow_out), Tensor(master_param_outs)"
    },
    "add": {
        "args": "Tensor x, Tensor y",
        "output": "Tensor(out)"
    },
    "add_n": {
        "args": "Tensor[] inputs",
        "output": "Tensor"
    },
    "affine_grid": {
        "args": "Tensor input, IntArray outputShape, bool align_corners=true",
        "output": "Tensor"
    },
    "all": {
        "args": "Tensor x, int64_t[] axis={}, bool keepdim=false",
        "output": "Tensor(out)"
    },
    "amax": {
        "args": "Tensor x, int64_t[] axis={}, bool keepdim=false",
        "output": "Tensor(out)"
    },
    "amin": {
        "args": "Tensor x, int64_t[] axis={}, bool keepdim=false",
        "output": "Tensor(out)"
    },
    "any": {
        "args": "Tensor x, int64_t[] axis={}, bool keepdim=false",
        "output": "Tensor(out)"
    },
    "arange": {
        "args": "Tensor start, Tensor end, Tensor step, DataType dtype, Place place={}",
        "output": "Tensor(out)"
    },
    "argmax": {
        "args": "Tensor x, Scalar axis, bool keepdims, bool flatten, int dtype",
        "output": "Tensor(out)"
    },
    "argmin": {
        "args": "Tensor x, Scalar axis, bool keepdims, bool flatten, int dtype",
        "output": "Tensor(out)"
    },
    "assign": {
        "args": "Tensor x",
        "output": "Tensor"
    },
    "assign_out_": {
        "args": "Tensor x, Tensor output",
        "output": "Tensor(out)"
    },
    "assign_value_": {
        "args": "Tensor output, int[] shape, DataType dtype, Scalar[] values, Place place = {}",
        "output": "Tensor(out)"
    },
    "auc": {
        "args": "Tensor x, Tensor label, Tensor stat_pos, Tensor stat_neg, Tensor ins_tag_weight, str curve, int num_thresholds, int slide_steps",
        "output": "Tensor(auc), Tensor(stat_pos_out), Tensor(stat_neg_out)"
    },
    "average_accumulates_": {
        "args": "Tensor param, Tensor in_sum_1, Tensor in_sum_2, Tensor in_sum_3, Tensor in_num_accumulates, Tensor in_old_num_accumulates, Tensor in_num_updates, float average_window, int64_t max_average_window, int64_t min_average_window",
        "output": "Tensor(out_sum_1), Tensor(out_sum_2), Tensor(out_sum_3), Tensor(out_num_accumulates), Tensor(out_old_num_accumulates), Tensor(out_num_updates)"
    },
    "batch_norm": {
        "args": "Tensor x, Tensor mean, Tensor variance, Tensor scale, Tensor bias, bool is_test, float momentum, float epsilon, str data_layout, bool use_global_stats, bool trainable_statistics",
        "output": "Tensor(out), Tensor(mean_out), Tensor(variance_out), Tensor(saved_mean), Tensor(saved_variance), Tensor(reserve_space)"
    },
    "bce_loss": {
        "args": "Tensor input, Tensor label",
        "output": "Tensor"
    },
    "bicubic_interp": {
        "args": "Tensor x, Tensor out_size, Tensor[] size_tensor, Tensor scale_tensor, str data_layout, int out_d, int out_h, int out_w, float[] scale, str interp_method, bool align_corners, int align_mode",
        "output": "Tensor(output)"
    },
    "bilinear_interp": {
        "args": "Tensor x, Tensor out_size, Tensor[] size_tensor, Tensor scale_tensor, str data_layout, int out_d, int out_h, int out_w, float[] scale, str interp_method, bool align_corners, int align_mode",
        "output": "Tensor(output)"
    },
    "bilinear_tensor_product": {
        "args": "Tensor x, Tensor y, Tensor weight, Tensor bias",
        "output": "Tensor"
    },
    "bincount": {
        "args": "Tensor x, Tensor weights, Scalar(int) minlength = 0",
        "output": "Tensor(out)"
    },
    "bitwise_and": {
        "args": "Tensor x, Tensor y",
        "output": "Tensor(out)"
    },
    "bitwise_not": {
        "args": "Tensor x",
        "output": "Tensor(out)"
    },
    "bitwise_or": {
        "args": "Tensor x, Tensor y",
        "output": "Tensor(out)"
    },
    "bitwise_xor": {
        "args": "Tensor x, Tensor y",
        "output": "Tensor(out)"
    },
    "box_coder": {
        "args": "Tensor prior_box, Tensor prior_box_var, Tensor target_box, str code_type, bool box_normalized, int axis, float[] variance",
        "output": "Tensor(output_box)"
    },
    "broadcast_tensors": {
        "args": "Tensor[] input",
        "output": "Tensor[]{input.size()}"
    },
    "cast": {
        "args": "Tensor x, DataType dtype",
        "output": "Tensor"
    },
    "channel_shuffle": {
        "args": "Tensor x, int groups, str data_format=\"NCHW\"",
        "output": "Tensor(out)"
    },
    "check_finite_and_unscale_": {
        "args": "Tensor[] x, Tensor scale, Tensor input_found_infinite",
        "output": "Tensor[](out){x.size()}, Tensor(output_found_infinite)"
    },
    "class_center_sample": {
        "args": "Tensor label, int num_classes, int num_samples, int ring_id, int rank, int nranks, bool fix_seed, int seed",
        "output": "Tensor(remapped_label), Tensor(sampled_local_class_center)"
    },
    "clip_by_norm": {
        "args": "Tensor x, float max_norm",
        "output": "Tensor(out)"
    },
    "coalesce_tensor": {
        "args": "Tensor[] input, DataType dtype, bool copy_data = false, bool set_constant = false, bool persist_output = false, float constant = 0.0, bool use_align = true, int align_size = -1, int size_of_dtype = -1, int64_t[] concated_shapes = {}, int64_t[] concated_ranks = {}",
        "output": "Tensor[](output){input.size()}, Tensor(fused_output)"
    },
    "concat": {
        "args": "Tensor[] x, Scalar(int64_t) axis",
        "output": "Tensor"
    },
    "conv2d": {
        "args": "Tensor input, Tensor filter, int[] strides, int[] paddings, str padding_algorithm, int[] dilations, int groups, str data_format",
        "output": "Tensor"
    },
    "conv2d_transpose": {
        "args": "Tensor x, Tensor filter, int[] strides, int[] paddings, int[] output_padding, IntArray output_size, str padding_algorithm, int groups, int[] dilations, str data_format",
        "output": "Tensor(out)"
    },
    "conv3d": {
        "args": "Tensor input, Tensor filter, int[] strides, int[] paddings, str padding_algorithm, int groups, int[] dilations, str data_format",
        "output": "Tensor"
    },
    "conv3d_transpose": {
        "args": "Tensor x, Tensor filter, int[] strides, int[] paddings, int[] output_padding, int[] output_size, str padding_algorithm, int groups, int[] dilations, str data_format",
        "output": "Tensor(out)"
    },
    "copy_to": {
        "args": "Tensor x, Place place, bool blocking",
        "output": "Tensor(out)"
    },
    "cross_entropy_with_softmax": {
        "args": "Tensor input, Tensor label, bool soft_label, bool use_softmax, bool numeric_stable_mode, int ignore_index, int axis",
        "output": "Tensor(softmax), Tensor(loss)"
    },
    "cumprod": {
        "args": "Tensor x,  int dim",
        "output": "Tensor(out)"
    },
    "cumsum": {
        "args": "Tensor x, Scalar axis, bool flatten, bool exclusive, bool reverse",
        "output": "Tensor(out)"
    },
    "decode_jpeg": {
        "args": "Tensor x, str mode, Place place",
        "output": "Tensor(out)"
    },
    "deformable_conv": {
        "args": "Tensor x, Tensor offset, Tensor filter, Tensor mask, int[] strides, int[] paddings, int[] dilations, int deformable_groups, int groups, int im2col_step",
        "output": "Tensor(out)"
    },
    "depthwise_conv2d": {
        "args": "Tensor x, Tensor filter, int[] strides, int[] paddings, str padding_algorithm, int groups, int[] dilations, str data_format",
        "output": "Tensor(out)"
    },
    "depthwise_conv2d_transpose": {
        "args": "Tensor x, Tensor filter, int[] strides, int[] paddings, int[] output_padding, IntArray output_size, str padding_algorithm, int groups, int[] dilations, str data_format",
        "output": "Tensor(out)"
    },
    "dirichlet": {
        "args": "Tensor alpha",
        "output": "Tensor(out)"
    },
    "distribute_fpn_proposals": {
        "args": "Tensor fpn_rois, Tensor rois_num, int min_level, int max_level, int refer_level, int refer_scale, bool pixel_offset",
        "output": "Tensor[](multi_fpn_rois){max_level - min_level + 1}, Tensor[](multi_level_rois_num){max_level - min_level + 1}, Tensor(restore_index)"
    },
    "divide": {
        "args": "Tensor x, Tensor y",
        "output": "Tensor"
    },
    "dropout": {
        "args": "Tensor x, Tensor seed_tensor, Scalar p, bool is_test, str mode, int seed, bool fix_seed",
        "output": "Tensor(out), Tensor(mask)"
    },
    "edit_distance": {
        "args": "Tensor hyps, Tensor refs, Tensor hypslength, Tensor refslength, bool normalized = false",
        "output": "Tensor(sequencenum), Tensor(out)"
    },
    "eigvalsh": {
        "args": "Tensor x, str uplo, bool is_test",
        "output": "Tensor(eigenvalues), Tensor(eigenvectors)"
    },
    "einsum": {
        "args": "Tensor[] x, str equation",
        "output": "Tensor, Tensor[]{x.size()}, Tensor[]{x.size()}"
    },
    "elementwise_heaviside": {
        "args": "Tensor x, Tensor y",
        "output": "Tensor"
    },
    "elementwise_pow": {
        "args": "Tensor x, Tensor y",
        "output": "Tensor(out)"
    },
    "embedding": {
        "args": "Tensor x, Tensor weight, int64_t padding_idx=-1, bool sparse=false",
        "output": "Tensor"
    },
    "empty": {
        "args": "IntArray shape, DataType dtype=DataType::FLOAT32, Place place=CPUPlace()",
        "output": "Tensor(out)"
    },
    "empty_like": {
        "args": "Tensor x, DataType dtype = DataType::UNDEFINED, Place place = {}",
        "output": "Tensor(out)"
    },
    "equal": {
        "args": "Tensor x, Tensor y",
        "output": "Tensor(out)"
    },
    "expand": {
        "args": "Tensor x, IntArray shape",
        "output": "Tensor"
    },
    "expand_as": {
        "args": "Tensor x, Tensor y, int[] target_shape",
        "output": "Tensor"
    },
    "exponential_": {
        "args": "Tensor x, float lam",
        "output": "Tensor(out)"
    },
    "eye": {
        "args": "Scalar num_rows, Scalar num_columns, DataType dtype=DataType::FLOAT32, Place place={}",
        "output": "Tensor(out)"
    },
    "fill": {
        "args": "Tensor x, Scalar value",
        "output": "Tensor(out)"
    },
    "flatten": {
        "args": "Tensor x, int start_axis, int stop_axis",
        "output": "Tensor(out), Tensor(xshape)"
    },
    "floor_divide": {
        "args": "Tensor x, Tensor y",
        "output": "Tensor(out)"
    },
    "fmax": {
        "args": "Tensor x, Tensor y",
        "output": "Tensor(out)"
    },
    "fmin": {
        "args": "Tensor x, Tensor y",
        "output": "Tensor(out)"
    },
    "frobenius_norm": {
        "args": "Tensor x, int64_t[] axis,  bool keep_dim,  bool reduce_all",
        "output": "Tensor(out)"
    },
    "full": {
        "args": "IntArray shape, Scalar value, DataType dtype=DataType::FLOAT32, Place place=CPUPlace()",
        "output": "Tensor(out)"
    },
    "full_": {
        "args": "Tensor output, IntArray shape, Scalar value, DataType dtype=DataType::FLOAT32, Place place=CPUPlace()",
        "output": "Tensor(out)"
    },
    "full_batch_size_like": {
        "args": "Tensor input, int[] shape, DataType dtype, Scalar value, int input_dim_idx, int output_dim_idx, Place place=CPUPlace()",
        "output": "Tensor(out)"
    },
    "full_like": {
        "args": "Tensor x, Scalar value, DataType dtype = DataType::UNDEFINED, Place place = {}",
        "output": "Tensor(out)"
    },
    "gather": {
        "args": "Tensor x, Tensor index, Scalar(int) axis=0",
        "output": "Tensor(out)"
    },
    "gaussian": {
        "args": "IntArray shape, float mean, float std, int seed, DataType dtype, Place place={}",
        "output": "Tensor(out)"
    },
    "generate_proposals": {
        "args": "Tensor scores, Tensor bbox_deltas, Tensor im_shape, Tensor anchors, Tensor variances, int pre_nms_top_n, int post_nms_top_n, float nms_thresh, float min_size, float eta, bool pixel_offset=true",
        "output": "Tensor(rpn_rois), Tensor(rpn_roi_probs), Tensor(rpn_rois_num)"
    },
    "greater_equal": {
        "args": "Tensor x, Tensor y",
        "output": "Tensor(out)"
    },
    "greater_than": {
        "args": "Tensor x, Tensor y",
        "output": "Tensor(out)"
    },
    "group_norm": {
        "args": "Tensor x, Tensor scale, Tensor bias, float epsilon, int groups, str data_layout",
        "output": "Tensor(y), Tensor(mean), Tensor(variance)"
    },
    "hardswish": {
        "args": "Tensor x",
        "output": "Tensor"
    },
    "hsigmoid_loss": {
        "args": "Tensor x, Tensor label, Tensor w, Tensor bias, Tensor path, Tensor code, int num_classes, bool remote_prefetch, bool is_sparse",
        "output": "Tensor(out), Tensor(pre_out), Tensor(w_out)"
    },
    "huber_loss": {
        "args": "Tensor input, Tensor label, float delta",
        "output": "Tensor(out), Tensor(residual)"
    },
    "increment": {
        "args": "Tensor x, float value = 1.0",
        "output": "Tensor(out)"
    },
    "index_add": {
        "args": "Tensor x, Tensor index,  Tensor add_value, int axis",
        "output": "Tensor(out)"
    },
    "instance_norm": {
        "args": "Tensor x, Tensor scale, Tensor bias, float epsilon",
        "output": "Tensor(y), Tensor(saved_mean), Tensor(saved_variance)"
    },
    "kldiv_loss": {
        "args": "Tensor x, Tensor label, str reduction",
        "output": "Tensor(out)"
    },
    "kron": {
        "args": "Tensor x, Tensor y",
        "output": "Tensor"
    },
    "lamb_": {
        "args": "Tensor param, Tensor grad, Tensor learning_rate, Tensor moment1, Tensor moment2, Tensor beta1_pow, Tensor beta2_pow, Tensor master_param, Tensor skip_update, float weight_decay, float beta1, float beta2, float epsilon, bool multi_precision",
        "output": "Tensor(param_out), Tensor(moment1_out), Tensor(moment2_out), Tensor(beta1_pow_out), Tensor(beta2_pow_out), Tensor(master_param_outs)"
    },
    "layer_norm": {
        "args": "Tensor x, Tensor scale, Tensor bias, float epsilon, int begin_norm_axis",
        "output": "Tensor(out), Tensor(mean), Tensor(variance)"
    },
    "less_equal": {
        "args": "Tensor x, Tensor y",
        "output": "Tensor(out)"
    },
    "less_than": {
        "args": "Tensor x, Tensor y",
        "output": "Tensor(out)"
    },
    "linear_interp": {
        "args": "Tensor x, Tensor out_size, Tensor[] size_tensor, Tensor scale_tensor, str data_layout, int out_d, int out_h, int out_w, float[] scale, str interp_method, bool align_corners, int align_mode",
        "output": "Tensor(output)"
    },
    "linspace": {
        "args": "Tensor start, Tensor stop, Tensor number, DataType dtype, Place place",
        "output": "Tensor(out)"
    },
    "log_softmax": {
        "args": "Tensor x,  int axis",
        "output": "Tensor(out)"
    },
    "logcumsumexp": {
        "args": "Tensor x, int axis, bool flatten, bool exclusive, bool reverse",
        "output": "Tensor(out)"
    },
    "logical_and": {
        "args": "Tensor x, Tensor y",
        "output": "Tensor(out)"
    },
    "logical_not": {
        "args": "Tensor x",
        "output": "Tensor(out)"
    },
    "logical_or": {
        "args": "Tensor x, Tensor y",
        "output": "Tensor(out)"
    },
    "logical_xor": {
        "args": "Tensor x, Tensor y",
        "output": "Tensor(out)"
    },
    "logsumexp": {
        "args": "Tensor x, int64_t[] axis,  bool keepdim,  bool reduce_all",
        "output": "Tensor(out)"
    },
    "lstsq": {
        "args": "Tensor x, Tensor y, Scalar rcond, str driver",
        "output": "Tensor(solution), Tensor(residuals), Tensor(rank), Tensor(singular_values)"
    },
    "lu": {
        "args": "Tensor x, bool pivot",
        "output": "Tensor(out), Tensor(pivots), Tensor(infos)"
    },
    "margin_cross_entropy": {
        "args": "Tensor logits, Tensor label, bool return_softmax, int ring_id, int rank, int nranks, float margin1, float margin2, float margin3, float scale",
        "output": "Tensor(softmax), Tensor(loss)"
    },
    "matmul": {
        "args": "Tensor x, Tensor y, bool transpose_x = false, bool transpose_y = false",
        "output": "Tensor"
    },
    "matrix_nms": {
        "args": "Tensor bboxes, Tensor scores, float score_threshold, int nms_top_k, int keep_top_k, float post_threshold=0., bool use_gaussian = false, float gaussian_sigma = 2.0, int background_label = 0, bool normalized = true",
        "output": "Tensor(out), Tensor(index), Tensor(roisnum)"
    },
    "matrix_rank": {
        "args": "Tensor x, float tol, bool hermitian=false, bool use_default_tol=true",
        "output": "Tensor(out)"
    },
    "matrix_rank_tol": {
        "args": "Tensor x, Tensor atol_tensor, bool use_default_tol=true, bool hermitian=false",
        "output": "Tensor(out)"
    },
    "max": {
        "args": "Tensor x, IntArray axis={}, bool keepdim=false",
        "output": "Tensor(out)"
    },
    "max_pool2d_with_index": {
        "args": "Tensor x, int[] kernel_size, int[] strides, int[] paddings, bool global_pooling, bool adaptive",
        "output": "Tensor(out), Tensor(mask)"
    },
    "max_pool3d_with_index": {
        "args": "Tensor x, int[] kernel_size, int[] strides, int[] paddings, bool global_pooling, bool adaptive",
        "output": "Tensor(out), Tensor(mask)"
    },
    "maximum": {
        "args": "Tensor x, Tensor y",
        "output": "Tensor(out)"
    },
    "mean": {
        "args": "Tensor x, IntArray axis={}, bool keepdim=false",
        "output": "Tensor(out)"
    },
    "mean_all": {
        "args": "Tensor x",
        "output": "Tensor"
    },
    "merge_selected_rows": {
        "args": "Tensor x",
        "output": "Tensor"
    },
    "merged_adam_": {
        "args": "Tensor[] param, Tensor[] grad, Tensor[] learning_rate, Tensor[] moment1, Tensor[] moment2, Tensor[] beta1_pow, Tensor[] beta2_pow, Tensor[] master_param, Scalar beta1, Scalar beta2, Scalar epsilon, bool multi_precision, bool use_global_beta_pow",
        "output": "Tensor[](param_out){param.size()}, Tensor[](moment1_out){param.size()}, Tensor[](moment2_out){param.size()}, Tensor[](beta1_pow_out){param.size()}, Tensor[](beta2_pow_out){param.size()}, Tensor[](master_param_out){param.size()}"
    },
    "merged_momentum_": {
        "args": "Tensor[] param, Tensor[] grad, Tensor[] velocity, Tensor[] learning_rate, Tensor[] master_param, float mu, bool use_nesterov = false, str[] regularization_method = {}, float[] regularization_coeff = {}, bool multi_precision = false, float rescale_grad = 1.0f",
        "output": "Tensor[](param_out){param.size()}, Tensor[](velocity_out){param.size()}, Tensor[](master_param_out){param.size()}"
    },
    "min": {
        "args": "Tensor x, IntArray axis={}, bool keepdim=false",
        "output": "Tensor(out)"
    },
    "minimum": {
        "args": "Tensor x, Tensor y",
        "output": "Tensor(out)"
    },
    "mish": {
        "args": "Tensor x, float lambda",
        "output": "Tensor"
    },
    "momentum_": {
        "args": "Tensor param, Tensor grad, Tensor velocity, Tensor learning_rate, Tensor master_param, float mu, bool use_nesterov = false, str regularization_method = \"\", float regularization_coeff = 0.0, bool multi_precision = false, float rescale_grad = 1.0f",
        "output": "Tensor(param_out), Tensor(velocity_out), Tensor(master_param_out)"
    },
    "multiclass_nms3": {
        "args": "Tensor bboxes, Tensor scores, Tensor rois_num, float score_threshold, int nms_top_k, int keep_top_k, float nms_threshold=0.3, bool normalized=true, float nms_eta=1.0, int background_label=0",
        "output": "Tensor(out), Tensor(index), Tensor(nms_rois_num)"
    },
    "multiply": {
        "args": "Tensor x, Tensor y",
        "output": "Tensor"
    },
    "nearest_interp": {
        "args": "Tensor x, Tensor out_size, Tensor[] size_tensor, Tensor scale_tensor, str data_layout, int out_d, int out_h, int out_w, float[] scale, str interp_method, bool align_corners, int align_mode",
        "output": "Tensor(output)"
    },
    "nms": {
        "args": "Tensor x, float threshold",
        "output": "Tensor(out)"
    },
    "nonzero": {
        "args": "Tensor condition",
        "output": "Tensor(out)"
    },
    "norm": {
        "args": "Tensor x, int axis, float epsilon, bool is_test",
        "output": "Tensor(out), Tensor(norm)"
    },
    "not_equal": {
        "args": "Tensor x, Tensor y",
        "output": "Tensor(out)"
    },
    "one_hot": {
        "args": "Tensor x, Scalar(int) num_classes",
        "output": "Tensor(out)"
    },
    "ones": {
        "args": "IntArray shape, DataType dtype=DataType::FLOAT32, Place place=CPUPlace()",
        "output": "Tensor(out)"
    },
    "ones_like": {
        "args": "Tensor x, DataType dtype=DataType::UNDEFINED, Place place={}",
        "output": "Tensor(out)"
    },
    "p_norm": {
        "args": "Tensor x,  float porder,  int axis,  float epsilon,  bool keepdim,  bool asvector=false",
        "output": "Tensor(out)"
    },
    "pad": {
        "args": "Tensor x, int[] paddings, Scalar pad_value",
        "output": "Tensor"
    },
    "pad3d": {
        "args": "Tensor x, IntArray paddings, str mode,  float pad_value, str data_format",
        "output": "Tensor(out)"
    },
    "pool2d": {
        "args": "Tensor x, IntArray kernel_size, int[] strides, int[] paddings, bool ceil_mode, bool exclusive, str data_format, str pooling_type, bool global_pooling, bool adaptive, str padding_algorithm",
        "output": "Tensor(out)"
    },
    "pool3d": {
        "args": "Tensor x, int[] kernel_size, int[] strides, int[] paddings, bool ceil_mode, bool exclusive, str data_format, str pooling_type, bool global_pooling, bool adaptive, str padding_algorithm",
        "output": "Tensor(out)"
    },
    "pow": {
        "args": "Tensor x, Scalar y",
        "output": "Tensor(out)"
    },
    "prelu": {
        "args": "Tensor x, Tensor alpha, str data_format, str mode",
        "output": "Tensor(out)"
    },
    "prior_box": {
        "args": "Tensor input, Tensor image, float[] min_sizes, float[] aspect_ratios, float[] variances, float[] max_sizes = {}, bool flip=true, bool clip=true, float step_w=0.0, float step_h=0.0, float offset=0.5, bool min_max_aspect_ratios_order=false",
        "output": "Tensor(out), Tensor(var)"
    },
    "prod": {
        "args": "Tensor x, IntArray dims, bool keep_dim, bool reduce_all",
        "output": "Tensor"
    },
    "psroi_pool": {
        "args": "Tensor x, Tensor boxes, Tensor boxes_num, int pooled_height, int pooled_width, int output_channels, float spatial_scale",
        "output": "Tensor"
    },
    "randint": {
        "args": "int low, int high, IntArray shape, DataType dtype=DataType::INT64, Place place={}",
        "output": "Tensor(out)"
    },
    "randperm": {
        "args": "int n, DataType dtype, Place place={}",
        "output": "Tensor(out)"
    },
    "relu6": {
        "args": "Tensor x",
        "output": "Tensor"
    },
    "remainder": {
        "args": "Tensor x, Tensor y",
        "output": "Tensor"
    },
    "repeat_interleave": {
        "args": "Tensor x, int repeats, int axis",
        "output": "Tensor(out)"
    },
    "repeat_interleave_with_tensor_index": {
        "args": "Tensor x, Tensor repeats, int axis",
        "output": "Tensor(out)"
    },
    "reshape": {
        "args": "Tensor x, IntArray shape",
        "output": "Tensor(out), Tensor(xshape)"
    },
    "reverse": {
        "args": "Tensor x, IntArray axis",
        "output": "Tensor"
    },
    "rmsprop_": {
        "args": "Tensor param, Tensor mean_square, Tensor grad, Tensor moment, Tensor learning_rate, Tensor mean_grad, float epsilon, float decay, float momentum, bool centered",
        "output": "Tensor(param_out), Tensor(moment_out), Tensor(mean_square_out), Tensor(mean_grad_out)"
    },
    "rnn": {
        "args": "Tensor x, Tensor[] pre_state, Tensor[] weight_list, Tensor sequence_length, Tensor dropout_state_in, float dropout_prob=0.0, bool is_bidirec=false, int input_size=10, int hidden_size=100, int num_layers=1, str mode=\"RNN_TANH\", int seed=0, bool is_test=false",
        "output": "Tensor(out), Tensor(dropout_state_out), Tensor[](state){pre_state.size()}, Tensor(reserve)"
    },
    "roi_align": {
        "args": "Tensor x, Tensor boxes, Tensor boxes_num, int pooled_height, int pooled_width, float spatial_scale, int sampling_ratio, bool aligned",
        "output": "Tensor"
    },
    "roi_pool": {
        "args": "Tensor x, Tensor boxes, Tensor boxes_num, int pooled_height, int pooled_width, float spatial_scale",
        "output": "Tensor(out), Tensor(arg_max)"
    },
    "rrelu": {
        "args": "Tensor x, float lower, float upper, bool is_test",
        "output": "Tensor(out), Tensor(noise)"
    },
    "scale": {
        "args": "Tensor x, Scalar scale, float bias, bool bias_after_scale",
        "output": "Tensor(out)"
    },
    "segment_pool": {
        "args": "Tensor x, Tensor segment_ids, str pooltype",
        "output": "Tensor(out), Tensor(summed_ids)"
    },
    "sgd_": {
        "args": "Tensor param, Tensor learning_rate, Tensor grad, Tensor master_param, bool multi_precision",
        "output": "Tensor(param_out), Tensor(master_param_out)"
    },
    "shape": {
        "args": "Tensor input",
        "output": "Tensor(out)"
    },
    "sigmoid_cross_entropy_with_logits": {
        "args": "Tensor x, Tensor label, bool normalize, int ignore_index",
        "output": "Tensor"
    },
    "sign": {
        "args": "Tensor x",
        "output": "Tensor(out)"
    },
    "slice": {
        "args": "Tensor input, int64_t[] axes, IntArray starts, IntArray ends, int64_t[] infer_flags, int64_t[] decrease_axis",
        "output": "Tensor"
    },
    "softmax": {
        "args": "Tensor x, int axis",
        "output": "Tensor(out)"
    },
    "spectral_norm": {
        "args": "Tensor weight, Tensor u, Tensor v, int dim, int power_iters, float eps",
        "output": "Tensor"
    },
    "split": {
        "args": "Tensor x, IntArray sections, Scalar(int) axis",
        "output": "Tensor[]{sections.size()}"
    },
    "split_with_num": {
        "args": "Tensor x, int num, Scalar(int) axis",
        "output": "Tensor[]{num}"
    },
    "squared_l2_norm": {
        "args": "Tensor x",
        "output": "Tensor"
    },
    "strided_slice": {
        "args": "Tensor x, int[] axes, IntArray starts, IntArray ends, IntArray strides",
        "output": "Tensor"
    },
    "subtract": {
        "args": "Tensor x, Tensor y",
        "output": "Tensor(out)"
    },
    "sum": {
        "args": "Tensor x, IntArray axis={}, DataType dtype=DataType::UNDEFINED, bool keepdim=false",
        "output": "Tensor(out)"
    },
    "swish": {
        "args": "Tensor x",
        "output": "Tensor(out)"
    },
    "sync_batch_norm_": {
        "args": "Tensor x, Tensor mean, Tensor variance, Tensor scale, Tensor bias, bool is_test, float momentum, float epsilon, str data_layout, bool use_global_stats, bool trainable_statistics",
        "output": "Tensor(out), Tensor(mean_out), Tensor(variance_out), Tensor(saved_mean), Tensor(saved_variance), Tensor(reserve_space)"
    },
    "temporal_shift": {
        "args": "Tensor x, int seg_num, float shift_ratio, str data_format_str",
        "output": "Tensor"
    },
    "tile": {
        "args": "Tensor x, IntArray repeat_times = {}",
        "output": "Tensor(out)"
    },
    "transpose": {
        "args": "Tensor x, int[] perm",
        "output": "Tensor"
    },
    "triangular_solve": {
        "args": "Tensor x, Tensor y, bool upper, bool transpose, bool unitriangular",
        "output": "Tensor"
    },
    "tril": {
        "args": "Tensor x,  int diagonal",
        "output": "Tensor(out)"
    },
    "tril_indices": {
        "args": "int rows, int cols, int offset, DataType dtype, Place place={}",
        "output": "Tensor(out)"
    },
    "trilinear_interp": {
        "args": "Tensor x, Tensor out_size, Tensor[] size_tensor, Tensor scale_tensor, str data_layout, int out_d, int out_h, int out_w, float[] scale, str interp_method, bool align_corners, int align_mode",
        "output": "Tensor(output)"
    },
    "triu": {
        "args": "Tensor x,  int diagonal",
        "output": "Tensor(out)"
    },
    "triu_indices": {
        "args": "int row, int col, int offset, DataType dtype, Place place={}",
        "output": "Tensor(out)"
    },
    "truncated_gaussian_random": {
        "args": "int[] shape, float mean, float std, int seed, DataType dtype=DataType::FLOAT32, Place place={}",
        "output": "Tensor(out)"
    },
    "uniform": {
        "args": "IntArray shape,  DataType dtype,  Scalar min,  Scalar max,  int seed, Place place={}",
        "output": "Tensor(out)"
    },
    "uniform_inplace": {
        "args": "Tensor x, float min, float max, int seed, int diag_num, int diag_step, float diag_val",
        "output": "Tensor(out)"
    },
    "unique": {
        "args": "Tensor x, bool return_index, bool return_inverse, bool return_counts, int[] axis, DataType dtype=DataType::INT64",
        "output": "Tensor(out), Tensor(indices), Tensor(inverse), Tensor(counts)"
    },
    "unpool": {
        "args": "Tensor x, Tensor indices, int[] ksize, int[] strides, int[] padding, IntArray output_size, str data_format",
        "output": "Tensor(out)"
    },
    "unpool3d": {
        "args": "Tensor x, Tensor indices, int[] ksize, int[] strides, int[] padding, int[] output_size, str data_format",
        "output": "Tensor(out)"
    },
    "update_loss_scaling_": {
        "args": "Tensor[] x, Tensor found_infinite, Tensor prev_loss_scaling, Tensor in_good_steps, Tensor in_bad_steps, int incr_every_n_steps, int decr_every_n_nan_or_inf, float incr_ratio, float decr_ratio, Scalar stop_update",
        "output": "Tensor[](out){x.size()}, Tensor(loss_scaling), Tensor(out_good_steps), Tensor(out_bad_steps)"
    },
    "warpctc": {
        "args": "Tensor logits, Tensor label, Tensor logits_length, Tensor labels_length, int blank, bool norm_by_times",
        "output": "Tensor(loss), Tensor(warpctcgrad)"
    },
    "yolo_box": {
        "args": "Tensor x, Tensor img_size, int[] anchors, int class_num, float conf_thresh, int downsample_ratio, bool clip_bbox, float scale_x_y=1.0, bool iou_aware=false, float iou_aware_factor=0.5",
        "output": "Tensor(boxes), Tensor(scores)"
    },
    "yolo_loss": {
        "args": "Tensor x, Tensor gt_box, Tensor gt_label, Tensor gt_score, int[] anchors, int[] anchor_mask, int class_num, float ignore_thresh, int downsample_ratio, bool use_label_smooth=true, float scale_x_y=1.0",
        "output": "Tensor(loss), Tensor(objectness_mask), Tensor(gt_match_mask)"
    },
    "zeros": {
        "args": "IntArray shape, DataType dtype=DataType::FLOAT32, Place place=CPUPlace()",
        "output": "Tensor(out)"
    },
    "zeros_like": {
        "args": "Tensor x, DataType dtype=DataType::UNDEFINED, Place place = {}",
        "output": "Tensor(out)"
    }
}
