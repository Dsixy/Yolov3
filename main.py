import torch
import torch.nn as nn
from util import Decoder, YoloLoss
from model import YOLOv3


if __name__ == "__main__":
    num_class = 20
    image_size = 416
    image = torch.randn(1, 3, image_size, image_size)
    net = YOLOv3(3, num_class)
    output1, output2, output3 = net(image)
    loss_fn = YoloLoss()

    assert output1.size() == (1, 75, 13, 13)
    assert output2.size() == (1, 75, 26, 26)
    assert output3.size() == (1, 75, 52, 52)
    outputs = [output1, output2, output3]
    optimizer = torch.optim.SGD(net.parameters(), lr=0.01)

    loss = loss_fn(outputs, torch.tensor([[[0.4, -0.6, 0.1, 0.3, 12], [-0.1, 0.2, 0.9, 0.4, 2]]]))
    # 反向传播
    optimizer.zero_grad()  # 梯度清零
    loss.backward()  # 反向传播，计算梯度

    # 更新模型参数
    optimizer.step()  # 更新参数
    print(loss)
    print("Success!")
