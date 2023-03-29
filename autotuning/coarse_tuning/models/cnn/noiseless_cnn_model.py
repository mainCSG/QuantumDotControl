import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from  matplotlib.colors import LinearSegmentedColormap
from matplotlib import colors


class CoarseTuningCNN(nn.Module):
    def __init__(self, image_size, num_classes):
        super(CoarseTuningCNN, self).__init__()

        self.image_size = image_size
        self.num_classes = num_classes

        # Layer 1
        self.conv1 = nn.Conv2d(1, 23, kernel_size=(5, 5), stride=2, padding=0, dtype=torch.float32)
        self.dropout1 = nn.Dropout(p=0.12)
        self.norm1 = nn.BatchNorm2d(23, dtype=torch.float32)

        # Layer 2
        self.conv2 = nn.Conv2d(23, 7, kernel_size=(5, 5), stride=2, padding=2, dtype=torch.float32)
        self.dropout2 = nn.Dropout(p=0.28)
        self.norm2 = nn.BatchNorm2d(7, dtype=torch.float32)

        # Layer 3
        self.conv3 = nn.Conv2d(7, 18, kernel_size=(5, 5), stride=2, padding=3, dtype=torch.float32)
        self.dropout3 = nn.Dropout(p=0.3)
        self.norm3 = nn.BatchNorm2d(18, dtype=torch.float32)

        # Average Pool + Softmax
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(18, num_classes, dtype=torch.float32)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # Layer 1
        x = F.relu(self.norm1(self.dropout1(self.conv1(x))))

        # Layer 2
        x = F.relu(self.norm2(self.dropout2(self.conv2(x))))

        # Layer 3
        x = F.relu(self.norm3(self.dropout3(self.conv3(x))))

        # Average Pool + Softmax
        x = self.avgpool(x)
        x = x.view(-1, 18)
        x = self.fc(x)
        x = self.softmax(x)
        return x
    

    def backprop(self,model, device, train_loader, optimizer, criterion):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):

            for i in range(data.shape[0]):
                image = data[i, :, :, :].unsqueeze(1)
                target_ = target[i,:,:]
                image, target_ = image.to(device), target_.to(device)

                optimizer.zero_grad()
                output = model(image)
                loss = criterion(output, target_)
                loss.backward()
                optimizer.step()

    def test(self,model, device, test_loader, criterion, voltages, tile_size):
        model.eval()
        test_loss = 0
        correct = 0
        counter = 0
        image_accuracy = []
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(test_loader):
                
                for i in range(data.shape[0]):
                    counter += 1

                    image = data[i, :, :, :].unsqueeze(1)
                    target_ = target[i,:,:]

                    image, target_ = image.to(device), target_.to(device)
                    output = model(image)

                    test_loss += criterion(output, target_).item() # sum up batch loss
                    pred = torch.argmax(output, dim=1, keepdim=True)

                    target_pred = torch.argmax(target_, dim=1, keepdim=True)

                    if counter == 1:
                        row_height = (self.image_size//tile_size)  # height of each row

                        predictions = np.array(pred.eq(target_pred).view(-1).tolist()).reshape(row_height,row_height)

                        # create a block with 1's and -1's based on the reshaped array
                        block = np.where(predictions == False, 1, -1)
                        temp = np.ones((tile_size,tile_size))
                        pred_img = np.kron(block,temp)
                        row_arrays = []
                        data = data.squeeze(0)
                        for i in range(0, data.shape[0], row_height):
                            tile = data[i:i+row_height,:,:].tolist()
                            row_arrays.append(np.concatenate(tile,axis=1))
                            np.concatenate(tile,axis=1).shape
                        reconstructed_image = np.concatenate(row_arrays, axis=0)
                        extent = [min(voltages[counter-1]["P1"]), max(voltages[counter-1]["P1"]),min(voltages[counter-1]["P2"]), max(voltages[counter-1]["P2"])]
                        plt.imshow(reconstructed_image, cmap='gray',extent=extent)
                        plt.imshow(pred_img,cmap = colors.ListedColormap(['green', 'red']), alpha=0.25,extent=extent)
                        plt.xlabel("P1")
                        plt.ylabel("P2")

                        plt.show()
                    # print(pred.shape, target_pred.shape)
                    correct += pred.eq(target_pred).sum().item()
                    # print(pred.eq(target_pred).sum().item(),(self.image_size//tile_size)**2 )
                # print(correct/ (self.image_size//tile_size)**2)
                image_accuracy.append(pred.eq(target_pred).sum().item()/ (self.image_size//tile_size)**2)
        # print(image_accuracy)
        test_loss /= len(test_loader)
        std_dev = np.array(image_accuracy).std()
        total = len(test_loader) * (self.image_size//tile_size)**2
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%), Standard Deviation: {:.3f}%\n'.format(
            test_loss, correct, total,
            100. * correct / total,100 * std_dev))
        
