# Nadav Levy 316409531 & Eran Deutsch 209191063

import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import wandb
from PIL import Image
from sklearn.metrics import confusion_matrix


class Cutout(object):
    def __init__(self, cutout_size=16):
        self.cutout_size = cutout_size

    def __call__(self, img):
        # Convert the PIL image to a NumPy array
        img = np.array(img)

        h, w, _ = img.shape
        x = np.random.randint(0, w - self.cutout_size)
        y = np.random.randint(0, h - self.cutout_size)

        # Apply cutout
        img[y:y + self.cutout_size, x:x + self.cutout_size, :] = 0

        # Convert the NumPy array back to a PIL image
        img = Image.fromarray(img)

        return img

class Net(nn.Module):
    # Creating the NN object
    def __init__(self , loss_type):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        if loss_type == "Base":
            self.fc3 = nn.Linear(84, 10)  # SoftMax
        else:
            self.fc3 = nn.Linear(84, 1)  # MSE

    def forward(self, x):
        #forward Propagation
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

 # Custom transform class to add noise
class AddNoise(object):
    def __init__(self, noise_level=0.1):
        self.noise_level = noise_level

    def __call__(self, x):
        noise = torch.randn_like(x) * self.noise_level
        return x + noise

args = {
         "lr": 0.01,
        "epochs": 2,                                                                    # number of times repeat the process on the training dataset
        "batch_size": 4,                                                               # number of training examples used in each iteration
        "weight_decay": 1e-4,                                                           # Weight decay for regularization
    }

transformations_names = ["Normal" , "With Noise" , "Position Augmantation" ,            # Transformation names
                         "Image Deformation" , "Color Augmantation"]
# Define a transform for loading CIFAR-10 data with noise
transformations = [
    transforms.Compose(                                                                 # Normalized
        [transforms.ToTensor(),transforms.Normalize
        ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),

    transforms.Compose([transforms.ToTensor(),                                          #Adding Noise
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    AddNoise(noise_level=0.1)]),

    transforms.Compose([
    transforms.RandomHorizontalFlip(),                                                  # Randomly flip images horizontally
    transforms.RandomRotation(10),                                                      # Randomly rotate images by up to 10 degrees
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),                           # Random affine transformation (translation)
    transforms.RandomResizedCrop(size=32, scale=(0.8, 1.0)),                            # Randomly crop and resize images
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),      # Color jitter
    transforms.ToTensor(),  # Convert to tensor
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),

    transforms.Compose([                                                                # Cutout Augmentation
        Cutout(),                                                                       # Apply Cutout
        transforms.ToTensor(),                                                          # Convert to tensor
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),

    transforms.Compose([
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Color jitter
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])]
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

PATH = './cifar_net.pth'
PATH_normal = './cifar_net_normal.pth'# save the model
loss_type_list = ["Base", "MSE", "L1"]


def loss_calc(decider):                                                                 # Activate different Loss func
    for i in range(len(loss_type_list)):
        augmantation(loss_type_list[i],decider)


def augmantation(loss_type,decider):
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transformations[0])
    testloader = torch.utils.data.DataLoader(testset, batch_size=args["batch_size"],
                                             shuffle=False, num_workers=2)

    if decider == 'A':
        for i in range(5):
            print("the current Augmentation " + transformations_names[i])
            trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                    download=True, transform=transformations[i])
            trainloader = torch.utils.data.DataLoader(trainset, batch_size=args["batch_size"],
                                                      shuffle=True, num_workers=2)

            train(trainloader,i,loss_type,True,PATH)
            test(testloader,loss_type)
    elif decider =='C':
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                download=True, transform=transformations[0])
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=args["batch_size"],
                                                  shuffle=True, num_workers=2)
        train(trainloader, 0, loss_type, True,PATH_normal)
        for i in range(1,5):

            print("the current Augmentation " + transformations_names[i])
            trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                    download=True, transform=transformations[i])
            trainloader = torch.utils.data.DataLoader(trainset, batch_size=args["batch_size"],
                                                      shuffle=True, num_workers=2)
            train(trainloader,i,loss_type,False,PATH)
            test(testloader,loss_type)
    else:
        print("the current Augmentation " + transformations_names[0])
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                                download=True, transform=transformations[0])
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=args["batch_size"],
                                                  shuffle=True, num_workers=2)
        train(trainloader, 0, loss_type,False,PATH)
        test(testloader, loss_type)

def validation(ToValidate,options):
    original = args[ToValidate]
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transformations[0])
    for i, parameter in enumerate(options):
        args[ToValidate] = parameter
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=args["batch_size"],
                                                  shuffle=True, num_workers=2)
        train_lost = train(trainloader, 0 , "Base", True, PATH,True,parameter)
        validation_loss = check(-1,"")
        print("Changing Argument {} to {} train loss: {}  validation loss {}.".format(ToValidate,parameter,train_lost,validation_loss))
    args[ToValidate] = original


def check(epoch,parameter):
    validationset = torchvision.datasets.CIFAR10('./data', train=False, transform=transformations[0],
                                                 download=True)
    validationloader = torch.utils.data.DataLoader(validationset, batch_size=args["batch_size"], shuffle=False)
    criterion = nn.CrossEntropyLoss()  # Softmax Loss
    net = Net("Base")
    net.load_state_dict(torch.load(PATH))
    name =  "accuracty per epoch validation {}".format(parameter)
    running_vloss = 0.0
    vtotal = 0
    vcorrect = 0
    running_vloss = 0.0
    with torch.no_grad():
        for i, vdata in enumerate(validationloader):
            vinputs, vlabels = vdata
            voutputs = net(vinputs)
            vloss = criterion(voutputs, vlabels)
            running_vloss += vloss
            _, vpredicted = torch.max(voutputs, 1)
            vtotal += vlabels.size(0)
            vcorrect += (vpredicted == vlabels).sum().item()
        if epoch != -1:
            wandb.log({name: 100 * vcorrect // vtotal})
    avg_vloss = running_vloss / (i + 1)

    return avg_vloss

def train(trainloader, index,loss_type, reset, dest, Acc2epoch,valid = ""):
    #wandb.login(key="e1a18173b7be434e1dc0fbaa98c3928e396accd8")
    # get some random training images
    dataiter = iter(trainloader)
    images, labels = next(dataiter)
    name = "accuracty per epoch train {}".format(valid)

    last_loss = 0
    total = 0
    correct = 0
    #show images
    imshow(torchvision.utils.make_grid(images))
    # print labels
    print(' '.join(f'{classes[labels[j]]:5s}' for j in range(args["batch_size"])))
    title = "loss " + transformations_names[index]+" "+loss_type
    net = Net(loss_type)
    if not reset:
        net.load_state_dict(torch.load(PATH_normal))

    wandb.watch(net, log_freq=100)
    if loss_type == "Base":
        criterion = nn.CrossEntropyLoss()  # Softmax Loss
    elif loss_type == "MSE":
        criterion = nn.MSELoss()  # MSE Loss
    else:
        criterion = nn.L1Loss()  # MAE Loss
    optimizer = optim.SGD(net.parameters(), lr=args["lr"], momentum=0.9, weight_decay=args["weight_decay"])
    for epoch in range(args["epochs"]):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            #print("iteration " + ((str)(i)) + " :")
            #print(outputs)
            #print(outputs.squeeze())
            #print(labels)
            if loss_type == "Base":
                loss = criterion(outputs, labels)  # Softmax
            else:
                loss = criterion(outputs.squeeze(), labels.float())  # MSE and  MAE
            #print(loss)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 200 == 199:  # print every 200 mini-batches
                last_loss = running_loss/200
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 200:.3f}')
                wandb.log({title: running_loss / 200})
                running_loss = 0.0
            if Acc2epoch:
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        if valid != "":
            torch.save(net.state_dict(), dest)
            check(epoch,valid)
            wandb.log({name: 100*correct//total},)


    print('Finished Training')
    torch.save(net.state_dict(), dest)
    return last_loss


def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def test(testloader,loss_type):

# check our model
    dataiter = iter(testloader)
    images, labels = next(dataiter)

    # print images
    #imshow(torchvision.utils.make_grid(images))
    print('GroundTruth: ', ' '.join(f'{classes[labels[j]]:5s}' for j in range(args["batch_size"])))
    net = Net(loss_type)
    net.load_state_dict(torch.load(PATH))
    outputs = net(images)
    #print(outputs)
    if loss_type == "Base":
        _, predicted = torch.max(outputs, 1)
    else:
        predicted = torch.round(outputs.squeeze())
        predicted = predicted.type(torch.int64).clamp(0,9)
    #print(predicted)

    print('Predicted: ', ' '.join(f'{classes[predicted[j]]:5s}'
                                  for j in range(args["batch_size"])))
    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            # calculate outputs by running images through the network
            outputs = net(images)
            # the class with the highest energy is what we choose as prediction
            if loss_type == "Base":
                _, predicted = torch.max(outputs, 1)
            else:
                predicted = torch.round(outputs.squeeze())
                predicted = predicted.type(torch.int64).clamp(0,9)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            #wandb.log({"conf_mat": wandb.plot.confusion_matrix(probs=None,y_true=labels, preds=predicted,class_names=['plane', 'car', 'bird', 'cat',
           #'deer', 'dog', 'frog', 'horse', 'ship', 'truck'])})

    print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')

    # prepare to count predictions for each class
    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}

    # again no gradients needed
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images)
            #_, predictions = torch.max(outputs, 1)
            if loss_type == "Base":
                _, predictions = torch.max(outputs, 1)
            else:
                predictions = torch.round(outputs.squeeze())
                predictions = predictions.type(torch.int64).clamp(0,9)
            # collect the correct predictions for each class
            for label, prediction in zip(labels, predictions):
                if label == prediction:
                    correct_pred[classes[label]] += 1
                total_pred[classes[label]] += 1

    # print accuracy for each class
    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[classname]
        print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')


if __name__ == '__main__':
    wandb.init(project="my-awesome-project", config=args)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transformations[0])
    testloader = torch.utils.data.DataLoader(testset, batch_size=args["batch_size"],
                                             shuffle=False, num_workers=2)

    #augmantation("Base", "A")                                                          # To run with different Image Augmantations with Softmax loss func
    #augmantation("Base", "C")                                                          # To run with different Image Augmantations with Softmax loss func
    #loss_calc('N')                                                                     # To run with All Loss func for original Images only
    #loss_calc('A')                                                                     # To run with all Loss func types and All Image augmantations
    #validation("epochs" , [2, 4, 16, 32, 64])                                          # Check Changes of Hyperparameters
    #test(testloader,"Base")
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
