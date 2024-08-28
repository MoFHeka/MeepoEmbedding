"""Proxy: Next Generation Polymorphism in C++."""

load("//:repositories.bzl", "proxy_repo")

def _proxy_dep_impl(_ctx):
    proxy_repo()

proxy_dep = module_extension(
    implementation = _proxy_dep_impl,
)
