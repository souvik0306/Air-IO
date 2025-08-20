import torch.nn as nn

class ONNXWrapper(nn.Module):
    """Expose explicit tensor inputs for ONNX export.

    The underlying network expects a dictionary of IMU tensors plus a rotation
    tensor and returns a dictionary containing ``net_vel`` and ``cov``.  ONNX
    does not support dictionary inputs or outputs, so this wrapper flattens the
    interface to three tensor arguments (``acc``, ``gyro``, ``rot``) and two
    tensor outputs (``net_vel``, ``cov``).
    """

    def __init__(self, net: nn.Module):
        super().__init__()
        self.net = net

    def forward(self, acc, gyro, rot):  # pragma: no cover - simple wrapper
        out = self.net({'acc': acc, 'gyro': gyro}, rot)
        return out['net_vel'], out['cov']

