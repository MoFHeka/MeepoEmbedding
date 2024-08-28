"""hkHierarchicalKV is a part of NVIDIA Merlin and provides hierarchical key-value storage to meet RecSys requirements. 
The key capability of HierarchicalKV is to store key-value feature-embeddings on high-bandwidth memory (HBM) of GPUs and in host memory. 
It also can be used as a generic key-value storage."""

load("//:repositories.bzl", "hkv_repo")

def _hkv_dep_impl(_ctx):
    hkv_repo()

hkv_dep = module_extension(
    implementation = _hkv_dep_impl,
)
