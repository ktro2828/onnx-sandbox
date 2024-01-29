import torch

custom_op_paths = ["build/lib.linux-x86_64-cpython-310/csrc/my_func/my_func.so"]

for path in custom_op_paths:
    torch.ops.load_library(path)


def test_add() -> None:
    # simple test
    print(">> Simple test")
    x = torch.ones(2, 2)
    y = torch.ones(2, 2)
    ans = torch.ops.my_func.add(x, y)
    print(f"{x}\n + {y}\n = {ans}")
    torch.testing.assert_close(ans, torch.ones(2, 2) * 2)

    # trace test
    print(">> Trace test")
    inputs = [x, y]
    trace = torch.jit.trace(add_trace, inputs)
    print(trace.graph)

    # script test
    print(">> Script test")
    print(add_script.graph)


def add_trace(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.ops.my_func.add(x, y)


@torch.jit.script
def add_script(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.ops.my_func.add(x, y)


def main():
    test_add()


if __name__ == "__main__":
    main()
