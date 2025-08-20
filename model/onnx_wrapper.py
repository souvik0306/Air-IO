import torch


class ONNXWrapper(torch.nn.Module):
    """Wrapper module for ONNX export.

    This module converts the network that expects a dictionary input into a
    module with explicit tensor arguments, which allows the exported ONNX graph
    to have named inputs.
    """

    def __init__(self, network: torch.nn.Module):
        super().__init__()
        self.network = network

    def forward(self, acc: torch.Tensor, gyro: torch.Tensor, rot: torch.Tensor):
        data = {"acc": acc, "gyro": gyro}
        out = self.network(data, rot)
        return out["net_vel"], out["cov"]
