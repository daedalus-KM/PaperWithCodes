import torch
import torch.nn as nn

class ConvBlock(nn.Module):
  def __init__(self, in_channel, out_channel, **kwargs):
    super(ConvBlock, self).__init__()
    self.conv = nn.Conv2d(in_channel, out_channel, **kwargs)
    self.batchnorm = nn.BatchNorm2d(out_channel)
    self.relu = nn.ReLU()

  def forward(self, x):
    out = self.conv(x)
    out = self.batchnorm(out)
    out = self.relu(out)
    return out
  
class Inception(nn.Module):
  def __init__(self, in_channel, n1x1, n3x3_reduce, n3x3, n5x5_reduce, n5x5, npool):
    super(Inception, self).__init__()

    self.branch1 = ConvBlock(in_channel, n1x1, kernel_size = 1, stride = 1, padding = 0)
    self.branch2 = nn.Sequential(
        ConvBlock(in_channel, n3x3_reduce, kernel_size = 1, stride = 1, padding = 0),
        ConvBlock(n3x3_reduce, n3x3, kernel_size = 3, stride = 1, padding = 1))
    self.branch3 = nn.Sequential(
        ConvBlock(in_channel, n5x5_reduce, kernel_size = 1, stride = 1, padding = 0),
        ConvBlock(n5x5_reduce, n5x5, kernel_size = 5, stride = 1, padding = 2))
    self.branch4 = nn.Sequential(
        nn.MaxPool2d(kernel_size = 3, stride = 1, padding = 1),
        ConvBlock(in_channel, npool, kernel_size = 1, stride = 1, padding = 0))

  def forward(self, x):
    x1 = self.branch1(x)
    x2 = self.branch2(x)
    x3 = self.branch3(x)
    x4 = self.branch4(x)
    return torch.cat([x1, x2, x3, x4], dim=1)
  
class Aux_Classifier(nn.Module):
  def __init__(self, in_channel, num_classes):
    super(Aux_Classifier, self).__init__()

    self.average_pool = nn.AvgPool2d(kernel_size = 5, stride = 2) #14 x 14 x channel -> 5x5xchannel
    self.conv = ConvBlock(in_channel, 128, kernel_size = 1, stride = 1, padding = 0)
    self.fc1 = nn.Linear(5 * 5 * 128, 1024)
    self.fc2 = nn.Linear(1024, num_classes)
    self.dropout = nn.Dropout(p=0.7)
    self.relu = nn.ReLU()

  def forward(self, x):
    out = self.average_pool(x)
    out = self.conv(out)
    out = out.view(out.size()[0], -1)
    out = self.fc1(out)
    out = self.dropout(out)
    out = self.relu(out)
    out = self.fc2(out)
    return out
  

class GoogLeNet(nn.Module):
  def __init__(self, aux = True, num_classes = 1000):
    super(GoogLeNet, self).__init__()
    assert aux == True or aux == False
    self.aux = aux

    self.conv1 = ConvBlock(1, 64, kernel_size = 7, stride = 2, padding = 3)
    self.pool1 = nn.MaxPool2d(kernel_size = 3 ,stride = 2, padding = 1)
    self.conv2 = ConvBlock(64, 64, kernel_size = 1, stride = 1, padding = 0)
    self.conv3 = ConvBlock(64, 192, kernel_size = 3, stride = 1, padding = 1)
    self.pool2 = nn.MaxPool2d(kernel_size = 3 ,stride = 2, padding = 1)

    self.inception_3a = Inception(192, 64, 96, 128, 16, 32, 32)
    self.inception_3b = Inception(256, 128, 128, 192, 32, 96, 64)
    self.pool3 = nn.MaxPool2d(kernel_size = 3 ,stride = 2, padding = 1)
    self.inception_4a = Inception(480, 192, 96, 208, 16, 48, 64)
    self.inception_4b = Inception(512, 160, 112, 224, 24, 64, 64)
    self.inception_4c = Inception(512, 128, 128, 256, 24, 64, 64)
    self.inception_4d = Inception(512, 112, 144, 288, 32, 64, 64)
    self.inception_4e = Inception(528, 256, 160, 320, 32, 128, 128)
    self.pool4 = nn.MaxPool2d(kernel_size = 3 ,stride = 2, padding = 1)
    self.inception_5a = Inception(832, 256, 160, 320, 32, 128, 128)
    self.inception_5b = Inception(832, 384, 192, 384, 48, 128, 128)
    self.avgpool = nn.AvgPool2d(kernel_size = 7, stride = 1)
    self.dropout = nn.Dropout(p=0.4)
    self.fc = nn.Linear(1024, num_classes)

    if self.aux:
      self.aux1 = Aux_Classifier(512, num_classes)
      self.aux2 = Aux_Classifier(528, num_classes)
    else:
      self.aux1 = None
      self.aux2 = None

  def forward(self, x):
    out = self.conv1(x)
    out = self.pool1(out)
    out = self.conv2(out)
    out = self.conv3(out)
    out = self.pool2(out)
    out = self.inception_3a(out)
    out = self.inception_3b(out)
    out = self.pool3(out)
    out = self.inception_4a(out)
    if self.aux and self.training:
      o_aux1 = self.aux1(out)
    out = self.inception_4b(out)
    out = self.inception_4c(out)
    out = self.inception_4d(out)
    if self.aux and self.training:
      o_aux2 = self.aux2(out)
    out = self.inception_4e(out)
    out = self.pool4(out)
    out = self.inception_5a(out)
    out = self.inception_5b(out)
    out = self.avgpool(out)
    out = out.view(out.size()[0], -1)
    out = self.fc(out)
    out = self.dropout(out)
    if self.aux and self.training:
      return o_aux1, o_aux2, out
    else:
      return out
    
#model summary
from torchsummary import summary
model1 = GoogLeNet()
summary(model1, input_size=(1, 224, 224), device = 'cpu')
# print(model)

#model test
x = torch.randn(3, 1, 224, 224).to('cpu')
output = model1(x)
print(output)

#prepare data
import numpy as np
from torch.utils.data import Dataset
from torchvision import datasets, transforms

transform = transforms.Compose([
    transforms.Resize(224),
    transforms.Grayscale(num_output_channels=1),  # 이미지를 흑백으로 변환
    transforms.ToTensor(),
])
train = datasets.FashionMNIST(root='data',
                              train=True,
                              download=True,
                              transform=transform
                             )

test = datasets.FashionMNIST(root='data',
                             train=False,
                             download=True,
                             transform=transform
                             )

train_loader = torch.utils.data.DataLoader(train, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test, batch_size=64, shuffle=False)

import matplotlib.pyplot as plt

labels_map = {
    0: "T-Shirt",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle Boot",
}

figure = plt.figure(figsize=(12, 8))
cols, rows = 8, 5

for i in range(1, cols * rows + 1):
    sample_idx = torch.randint(len(train), size=(1,)).item()
    img, label = train[sample_idx]
    figure.add_subplot(rows, cols, i)
    plt.title(labels_map[label])
    plt.axis("off")
    plt.imshow(torch.permute(img, (1, 2, 0)), cmap='gray')
plt.show()

#train
if __name__ == "__main__":
    # set hyperparameter

    np.random.seed(1)
    seed = torch.manual_seed(1)

    # model, loss, optimizer
    model = GoogLeNet(num_classes=10)
    device = torch.device("cpu")
    model.to(device)
    losses = []
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0002)

    # train
    for epoch in range(10):
        model.train()
        train_loss = 0
        correct, count = 0, 0
        for batch_idx, (images, labels) in enumerate(train_loader, start=1):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            output = model.forward(images)
            aux1, aux2, x = output
            loss = criterion(x, labels) + 0.3 * (criterion(aux1, labels) + criterion(aux2, labels))
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, preds = torch.max(x, 1)
            count += labels.size(0)
            correct += preds.eq(labels).sum().item() # torch.sum(preds == labels)

            if batch_idx % 5 == 0:
                print (f"[*] Epoch: {epoch} \tStep: {batch_idx}/{len(train_loader)}\tTrain accuracy: {round((correct/count), 4)} \tTrain Loss: {round((train_loss/count)*100, 4)}")

        # valid
        model.eval()
        correct, count = 0, 0
        valid_loss = 0
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(test_loader, start=1):
                images, labels = images.to(device), labels.to(device)
                output = model.forward(images)
                loss = criterion(output, labels)
                valid_loss += loss.item()
                _, preds = torch.max(output, 1)
                count += labels.size(0)
                correct += preds.eq(labels).sum().item() # torch.sum(preds == labels)
                if batch_idx % 100 == 0:
                    print (f"[*] Step: {batch_idx}/{len(test_loader)}\tValid accuracy: {round((correct/count), 4)} \tValid Loss: {round((valid_loss/count)*100, 4)}")

        # if epoch % 10 == 0:
        #     if not os.path.isdir('../checkpoint'):
        #         os.makedirs('../checkpoint', exists_ok=True)
        #     checkpoint_path = os.path.join(f"../checkpoint/googLeNet{epoch}.pth")
        #     state = {
        #         'epoch': epoch,
        #         'model': model.state_dict(),
        #         'optimizer': optimizer.state_dict(),
        #         'seed': seed,
        #     }
        #     torch.save(state, checkpoint_path)