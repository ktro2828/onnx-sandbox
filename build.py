import os.path as osp

from torch.utils.cpp_extension import BuildExtension, CUDAExtension


def make_cuda_ext(
    name: str,
    module: str,
    sources: list[str],
    cxx_std: int = 17,
) -> CUDAExtension:
    assert cxx_std in (17, 20), f"Unexpected CXX standard: {cxx_std}"
    cuda_ext = CUDAExtension(
        name=f"{module}.{name}",
        sources=[osp.join(*module.split("."), src) for src in sources],
        extra_compile_args=[f"-std=c++{cxx_std}", "-v"],
    )
    return cuda_ext


def build(setup_kwargs: dict) -> None:
    ext_modules = [
        make_cuda_ext(
            name="my_func",
            module="csrc.my_func",
            sources=["add.cpp"],
        ),
        make_cuda_ext(
            name="my_autograd",
            module="csrc.my_autograd",
            sources=[
                "attention_kernel.cpp",
                "attention_value_kernel.cu",
                "attention_weight_kernel.cu",
                "my_attention.cpp",
            ],
        ),
    ]

    setup_kwargs.update(
        {
            "ext_modules": ext_modules,
            "cmdclass": {
                "build_ext": BuildExtension.with_options(no_python_abi_suffix=True),
            },
        }
    )
