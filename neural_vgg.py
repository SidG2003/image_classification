import os
import torch
import torchvision
import tarfile
from torchvision.datasets.utils import download_url
from torch.utils.data import random_split

from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor

import matplotlib
import matplotlib.pyplot as plt
# %matplotlib inline

from torch.utils.data.dataloader import DataLoader

from torchvision.utils import make_grid

import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

from ImageClassificationBase import ImageClassificationBase, accuracy
from model import Cifar10CnnModel, get_model

project_name='05-vgg-cnn'

# Dowload the dataset
# dataset_url = "https://s3.amazonaws.com/fast-ai-imageclas/cifar10.tgz"
# download_url(dataset_url, '.')

# Extract from archive
# with tarfile.open('./cifar10.tgz', 'r:gz') as tar:
#     tar.extractall(path='./data')

data_dir = './data/awa2'

print(os.listdir(data_dir))
classes = os.listdir(data_dir + "/train")
print(classes)

# airplane_files = os.listdir(data_dir + "/train/airplane")
# print('No. of training examples for airplanes:', len(airplane_files))
# print(airplane_files[:5])

# ship_test_files = os.listdir(data_dir + "/test/ship")
# print("No. of test examples for ship:", len(ship_test_files))
# print(ship_test_files[:5])


normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

dataset = ImageFolder(
    data_dir+'/train', 
    transform=transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

img, label = dataset[0]
print(img.shape, label)
# print(img)

print(dataset.classes)

matplotlib.rcParams['figure.facecolor'] = '#ffffff'

def show_example(img, label):
    print('Label: ', dataset.classes[label], "("+str(label)+")")
    plt.imshow(img.permute(1, 2, 0))
    plt.show()

# show_example(*dataset[0])

random_seed = 42
torch.manual_seed(random_seed)

# val_size = 5000
# train_size = len(dataset) - val_size


train_size = int(0.8*len(dataset))
val_size = len(dataset)-train_size

train_ds, val_ds = random_split(dataset, [train_size, val_size])
print(len(train_ds), len(val_ds))

batch_size=64

train_dl = DataLoader(train_ds, 
                      batch_size, 
                      shuffle=True, 
                    #   num_workers=4, 
                      pin_memory=True)

val_dl = DataLoader(val_ds, 
                    batch_size*2, 
                    # num_workers=4, 
                    pin_memory=True)

def show_batch(dl):
    for images, labels in dl:
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.set_xticks([]); ax.set_yticks([])
        ax.imshow(make_grid(images, nrow=16).permute(1, 2, 0))
        plt.show()
        break

# show_batch(train_dl)
    
# model = Cifar10CnnModel()
# print(model)

model = get_model('vgg16', len(classes), pretrained=True)
print(model)

def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')
    
def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

class DeviceDataLoader():
    """Wrap a dataloader to move data to a device"""
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device
        
    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl: 
            yield to_device(b, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.dl)
    
device = get_default_device()
print(device)

train_dl = DeviceDataLoader(train_dl, device)
val_dl = DeviceDataLoader(val_dl, device)
to_device(model, device)

print("-----model is pn cuda:",next(model.parameters()).is_cuda)


# train model
@torch.no_grad()
def evaluate(model, val_loader):
    model.eval()
    outputs=[]
    # batch_no=0
    for  batch in val_loader:
        images, labels = batch 
        out = model(images)                    # Generate predictions
        loss = F.cross_entropy(out, labels)   # Calculate loss
        acc = accuracy(out, labels)           # Calculate accuracy
        outputs.append({'val_loss': loss.detach(), 'val_acc': acc})
        # outputs.append(model.validation_step(batch))
        
    # outputs = [model.validation_step(batch) for batch in val_loader]
    batch_losses = [x['val_loss'] for x in outputs]
    epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
    batch_accs = [x['val_acc'] for x in outputs]
    epoch_acc = torch.stack(batch_accs).mean()      # Combine accuracies
    return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}
    # return model.validation_epoch_end(outputs)

def fit(epochs, lr, model, train_loader, val_loader, opt_func=torch.optim.SGD):
    history = []
    optimizer = opt_func(model.parameters(), lr)
    epoch_num=0
    for epoch in range(epochs):
        print("ep num",epoch_num)
        # Training Phase 
        model.train()
        train_losses = []
        batch_num=0
        for batch in train_loader:
            print("batch: " ,batch_num)
            # loss = model.training_step(batch)
            images, labels = batch 
            out = model(images)                  # Generate predictions
            loss = F.cross_entropy(out, labels) # Calculate loss
            train_losses.append(loss)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            batch_num+=1
            del batch, loss,images, labels
        # Validation phase
        result = evaluate(model, val_loader)
        result['train_loss'] = torch.stack(train_losses).mean().item()
        # model.epoch_end(epoch, result)
        print("Epoch [{}], train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
            epoch, result['train_loss'], result['val_loss'], result['val_acc']))
        history.append(result)
        print("result: " , result)
        epoch_num+=1
    return history

# model = to_device(Cifar10CnnModel(), device)
print("accuracy_before_training:")
print(evaluate(model, val_dl))

num_epochs = 2
opt_func = torch.optim.Adam
lr = 0.001

history = fit(num_epochs, lr, model, train_dl, val_dl, opt_func)

def plot_accuracies(history):
    accuracies = [x['val_acc'] for x in history]
    plt.plot(accuracies, '-x')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.title('Accuracy vs. No. of epochs')
    plt.show()

plot_accuracies(history)

def plot_losses(history):
    train_losses = [x.get('train_loss') for x in history]
    val_losses = [x['val_loss'] for x in history]
    plt.plot(train_losses, '-bx')
    plt.plot(val_losses, '-rx')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['Training', 'Validation'])
    plt.title('Loss vs. No. of epochs')
    plt.show()

plot_losses(history)

# individual img test
# test_dataset = ImageFolder(data_dir+'/test', transform=ToTensor())
test_dataset = ImageFolder(
    data_dir+'/test', 
    transform=transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))



def predict_image(img, model):
    # Convert to a batch of 1
    xb = to_device(img.unsqueeze(0), device)
    # Get predictions from model
    yb = model(xb)
    # Pick index with highest probability
    _, preds  = torch.max(yb, dim=1)
    # Retrieve the class label
    return dataset.classes[preds[0].item()]

img, label = test_dataset[0]
plt.imshow(img.permute(1, 2, 0))
print('Label:', dataset.classes[label], ', Predicted:', predict_image(img, model))

test_loader = DeviceDataLoader(DataLoader(test_dataset, batch_size*2), device)
result = evaluate(model, test_loader)
print(result)

torch.save(model.state_dict(), 'vgg16-cnn.pth')
model2 = to_device(Cifar10CnnModel(), device)
model2.load_state_dict(torch.load('vgg16-cnn.pth'))

print(evaluate(model2, test_loader))

