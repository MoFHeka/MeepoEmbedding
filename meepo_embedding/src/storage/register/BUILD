load("//meepo_embedding/include:dep_rules.bzl", "storage_deps")

package(
    licenses = ["notice"],
)

cc_library(
    name = "storage_register",
    srcs = glob(
        ["*.cpp"],
        exclude = ["*_test.cpp"],
    ),
    hdrs = ["//meepo_embedding/include/storage/register:headers"],
    visibility = ["//visibility:public"],
    deps = [
        "//meepo_embedding/src/common",
    ] + storage_deps(),
)

cc_test(
    name = "storage_registry_test",
    size = "small",
    srcs = ["storage_registry_test.cpp"],
    copts = ["-isystem external/googletest~/include"],
    deps = [
        ":storage_register",
        "@googletest//:gtest",
        "@googletest//:gtest_main",
    ] + storage_deps(),
)
