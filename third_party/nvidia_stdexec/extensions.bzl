"""stdexec: A Standard Model for Asynchronous Execution in C++."""

load("//third_party/nvidia_stdexec:repositories.bzl", "nvidia_stdexec_repo")

def _stdexec_dep_impl(_ctx):
    nvidia_stdexec_repo()

nvidia_stdexec_dep = module_extension(
    implementation = _stdexec_dep_impl,
)
