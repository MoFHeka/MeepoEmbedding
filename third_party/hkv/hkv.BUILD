"""hkHierarchicalKV is a part of NVIDIA Merlin and provides hierarchical key-value storage to meet RecSys requirements. 
The key capability of HierarchicalKV is to store key-value feature-embeddings on high-bandwidth memory (HBM) of GPUs and in host memory. 
It also can be used as a generic key-value storage."""

load("@rules_cuda//cuda:defs.bzl", "cuda_library")

licenses(["notice"])  # Apache 2

exports_files(["LICENSE"])

cuda_library(
    name = "hkv",
    hdrs = glob([
        "include/merlin/core_kernels/*.cuh",
        "include/merlin/*.cuh",
        "include/*.cuh",
        "include/*.hpp",
    ]),
    copts = [
        "-Ofast",
    ],
    include_prefix = "include",
    includes = ["include"],
    visibility = ["//visibility:public"],
)
