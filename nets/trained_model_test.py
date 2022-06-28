import torch
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import os
from model import CNN


device_type = "GPU" if torch.cuda.is_available() else "CPU"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("using {} device.".format(device))

transform = transforms.Compose([transforms.Resize([224, 224]),
                                transforms.Grayscale(num_output_channels=1),
                                transforms.ToTensor(),
                                ])

# 载入自己的数据集
image_path = '../dataset/stock_data'
test_dataset = datasets.ImageFolder(root=os.path.join(image_path, "test"), transform=transform)
test_loader = DataLoader(test_dataset, shuffle=False, batch_size=64)

# 生成卷积神经网络并载入训练好的模型
model = CNN()
model.load_state_dict(torch.load("model_weight_cuda.pth".format(device_type)))


def test():
    correct = 0
    total = 0
    print("label       predicted")
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs.data, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()


        print('CNN trained model： accuracy on my_mnist_dataset set:%d %%' % (100 * correct / total))


if __name__ == '__main__':
    test()
