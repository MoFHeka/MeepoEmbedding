package(
    licenses = ["notice"],
)

cc_library(
    name = "common",
    srcs = glob(["*.cpp"]),
    hdrs = ["//meepo_embedding/include/common:headers"],
    visibility = ["//visibility:public"],
    deps = [
        "@dlpack",
        "@magic_enum",
        "@quill",
    ],
)
