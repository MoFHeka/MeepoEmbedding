"""Proxy: Next Generation Polymorphism in C++."""

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive", "new_git_repository")

def proxy_repo():
    # http_archive(
    #     name = "proxy",
    #     sha256 = "7eed973655938d681a90dcc0c200e6cc1330ea8611a9c1a9e1b30439514443cb",
    #     strip_prefix = "proxy-2.4.0",
    #     url = "https://github.com/microsoft/proxy/archive/refs/tags/2.4.0.tar.gz",
    #     build_file = "//third_party/proxy:proxy.BUILD",
    # )
    new_git_repository(
        name = "proxy",
        remote = "https://github.com/microsoft/proxy.git",
        branch = "main",
        build_file = "//third_party/proxy:proxy.BUILD",
    )
