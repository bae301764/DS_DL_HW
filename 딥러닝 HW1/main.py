
# import dataset
from model import LeNet5, CustomMLP, LeNet5_advance
from dataset import MNIST
from torch.utils.data import DataLoader
from multiprocessing import cpu_count
import torch
from torchsummary import summary
import matplotlib.pyplot as plt
from tqdm import tqdm
# import some packages you need here

def train(model, trn_loader, device, criterion, optimizer):
    """ Train function

    Args:
        model: network
        trn_loader: torch.utils.data.DataLoader instance for training
        device: device for computing, cpu or gpu
        criterion: cost function
        optimizer: optimization method, refer to torch.optim

    Returns:
        trn_loss: average loss value
        acc: accuracy
    """
    trn_loss = 0
    batch_accu = 0
    total = 0
    model.train()
    active = torch.nn.Softmax(dim=1)
    for img, label in trn_loader:
        img = img.to(device)
        label = label.to(device)

        optimizer.zero_grad()
        logit = model(img)
        prob = active(logit)
        loss = criterion(prob, label)
        loss.backward()
        optimizer.step()

        trn_loss += loss.item()
        pred = torch.argmax(prob, axis=1)
        batch_accu += (pred == label).sum().item()
        total += len(label)

    trn_loss = trn_loss/len(trn_loader)
    acc = batch_accu / total
    
    return trn_loss, acc

def test(model, tst_loader, device, criterion):
    """ Test function

    Args:
        model: network
        tst_loader: torch.utils.data.DataLoader instance for testing
        device: device for computing, cpu or gpu
        criterion: cost function

    Returns:
        tst_loss: average loss value
        acc: accuracy
    """
    with torch.no_grad():
        model.eval()
        tst_loss = 0
        batch_accu = 0
        total = 0
        active = torch.nn.Softmax(dim=1)
        for img, label in tst_loader:
            img = img.to(device)
            label = label.to(device)
            logit = model(img)
            
            prob = active(logit)
            loss = criterion(prob, label)

            tst_loss += loss.item()
            pred = torch.argmax(prob, axis=1)
            batch_accu += (pred == label).sum().item()
            total += len(label)
        tst_loss = tst_loss/len(tst_loader)
        acc = batch_accu / total
    
    return tst_loss, acc


def main():
    """ Main function

        Here, you should instantiate
        1) Dataset objects for training and test datasets
        2) DataLoaders for training and testing
        3) model
        4) optimizer: SGD with initial learning rate 0.01 and momentum 0.9
        5) cost function: use torch.nn.CrossEntropyLoss

    """

    device = torch.device('cuda')


    # 1) Dataset objects for training and test datasets
    train_data_dir = '/home/iai3/Desktop/jeongwon/2024 딥러닝 과제/mnist-classification/data/train'
    test_data_dir = '/home/iai3/Desktop/jeongwon/2024 딥러닝 과제/mnist-classification/data/test'

    train_set = MNIST(train_data_dir)
    test_set = MNIST(test_data_dir)
    
    # 2) DataLoaders for training and test
    batch_size = 512
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=16)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=16)

    # 3) model
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    Lenet = LeNet5().to(device)

    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    Lenet_advance = LeNet5_advance().to(device)

    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    customMLP = CustomMLP().to(device)
    
    print('LeNet-5 구조')
    print(summary(Lenet, (1, 28, 28)))
    print('custom MLP 구조')
    print(summary(customMLP, (1, 28, 28)))

    # 4) optimizer
    optimizer_lenet = torch.optim.SGD(Lenet.parameters(), lr=0.01, momentum=0.9)
    optimizer_advance = torch.optim.SGD(Lenet_advance.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-3)
    optimizer_mlp = torch.optim.SGD(customMLP.parameters(), lr=0.01, momentum=0.9)

    # 5) cost function
    loss_lenet = torch.nn.CrossEntropyLoss()
    loss_advance = torch.nn.CrossEntropyLoss()
    loss_mlp = torch.nn.CrossEntropyLoss()

    # 6) Train
    epochs = 30

    train_loss_list_lenet, train_accuracy_list_lenet, test_loss_list_lenet, test_accuracy_list_lenet = [], [], [], []
    train_loss_list_advance, train_accuracy_list_advance, test_loss_list_advance, test_accuracy_list_advance = [], [], [], []
    train_loss_list_mlp, train_accuracy_list_mlp, test_loss_list_mlp, test_accuracy_list_mlp = [], [], [], []

    print('----- Training LeNet')
    for _ in tqdm(range(epochs)):
        # Train
        train_loss_lenet, train_accuracy_lenet = train(Lenet, train_loader, device, loss_lenet, optimizer_lenet)
        train_loss_list_lenet.append(train_loss_lenet)
        train_accuracy_list_lenet.append(train_accuracy_lenet)
        # Test
        test_loss_lenet, test_accuracy_lenet = test(Lenet, test_loader, device, loss_lenet)
        test_loss_list_lenet.append(test_loss_lenet)
        test_accuracy_list_lenet.append(test_accuracy_lenet)

    print('----- Training LeNet-Advance')
    for _ in tqdm(range(epochs)):
        # Train
        train_loss_advance, train_accuracy_advance = train(Lenet_advance, train_loader, device, loss_advance, optimizer_advance)
        train_loss_list_advance.append(train_loss_advance)
        train_accuracy_list_advance.append(train_accuracy_advance)
        # Test
        test_loss_advance, test_accuracy_advance = test(Lenet_advance, test_loader, device, loss_advance)
        test_loss_list_advance.append(test_loss_advance)
        test_accuracy_list_advance.append(test_accuracy_advance)

    print('----- Training CustomMLP')
    for _ in tqdm(range(epochs)):
        # Train
        train_loss_mlp, train_accuracy_mlp = train(customMLP, train_loader, device, loss_mlp, optimizer_mlp)
        train_loss_list_mlp.append(train_loss_mlp)
        train_accuracy_list_mlp.append(train_accuracy_mlp)
        # Test
        test_loss_mlp, test_accuracy_mlp = test(customMLP, test_loader, device, loss_mlp)
        test_loss_list_mlp.append(test_loss_mlp)
        test_accuracy_list_mlp.append(test_accuracy_mlp)

    fig, axe = plt.subplots(2,2,figsize=(17,15))
    axe[0,0].plot(train_loss_list_lenet)
    axe[0,0].plot(train_loss_list_advance)
    axe[0,0].plot(train_loss_list_mlp)
    axe[0,0].legend(['LeNet', 'LeNet_advance', 'CustomMLP'])
    axe[0,0].set_title('Train Loss')


    axe[0,1].plot(train_accuracy_list_lenet)
    axe[0,1].plot(train_accuracy_list_advance)
    axe[0,1].plot(train_accuracy_list_mlp)
    axe[0,1].legend(['LeNet', 'LeNet_advance', 'CustomMLP'])
    axe[0,1].set_title('Train Accuracy')
        
    axe[1,0].plot(test_loss_list_lenet)
    axe[1,0].plot(test_loss_list_advance)
    axe[1,0].plot(test_loss_list_mlp)
    axe[1,0].legend(['LeNet', 'LeNet_advance', 'CustomMLP'])
    axe[1,0].set_title('Test Loss')

    axe[1,1].plot(test_accuracy_list_lenet)
    axe[1,1].plot(test_accuracy_list_advance)
    axe[1,1].plot(test_accuracy_list_mlp)
    axe[1,1].legend(['LeNet', 'LeNet_advance', 'CustomMLP'])
    axe[1,1].set_title('Test Accuracy')
    
    axe[1,1].text(len(test_accuracy_list_lenet), 0.6, f'LeNet : {test_accuracy_list_lenet[-1]:.4f}', ha='right')
    axe[1,1].text(len(test_accuracy_list_advance), 0.55, f'LeNet Advance: {test_accuracy_list_advance[-1]:.4f}', ha='right')
    axe[1,1].text(len(test_accuracy_list_mlp), 0.5, f'MLP: {test_accuracy_list_mlp[-1]:.4f}', ha='right', va='bottom')

    fig.tight_layout()
    fig.savefig('./loss and accuracy plot2.png')


if __name__ == '__main__':
    main()
