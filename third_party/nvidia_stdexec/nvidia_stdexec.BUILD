# Description:
#   stdexec: A Standard Model for Asynchronous Execution in C++.

licenses(["notice"])  # Apache 2

exports_files(["LICENSE"])

cc_library(
    name = "stdexec",
    hdrs = glob([
        "include/stdexec/**/*.hpp",
        "include/exec/**/*.hpp",
    ]),
    includes = ["include"],
    linkopts = ["-pthread"],
    visibility = ["//visibility:public"],
)
