"""stdexec: Next Generation Polymorphism in C++."""

load("@bazel_tools//tools/build_defs/repo:git.bzl", "new_git_repository")

def nvidia_stdexec_repo():
    new_git_repository(
        name = "stdexec",
        remote = "https://github.com/NVIDIA/stdexec.git",
        branch = "main",
        build_file = "//third_party/nvidia_stdexec:BUILD",
    )
