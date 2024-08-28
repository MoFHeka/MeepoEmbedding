"""hkHierarchicalKV is a part of NVIDIA Merlin and provides hierarchical key-value storage to meet RecSys requirements. 
The key capability of HierarchicalKV is to store key-value feature-embeddings on high-bandwidth memory (HBM) of GPUs and in host memory. 
It also can be used as a generic key-value storage."""

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

def hkv_repo():
    http_archive(
        name = "hkv",
        sha256 = "a73d7bea159173db2038f7c5215a7d1fbd5362adfb232fabde206dc64a1e817c",
        strip_prefix = "hkv-0.1.0-beta.12",
        url = "https://github.com/NVIDIA-Merlin/HierarchicalKV/archive/refs/tags/v0.1.0-beta.12.tar.gz",
        build_file = "//third_party/hkv:hkv.BUILD",
    )
