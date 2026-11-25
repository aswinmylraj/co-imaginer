import torch


def test_cuda():
    assert torch.cuda.is_available(), (
        "PyTorch sees no GPU. nvidia-smi must show driver 570 or newer."
    )
    assert (torch.ones(2, device="cuda") * 2).sum().item() == 4.0
