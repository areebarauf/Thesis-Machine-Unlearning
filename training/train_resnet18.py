import os
import time
import model as m
import torch
from torch import nn
import torch.nn.functional as F
import torchvision.transforms as tt
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision.models import resnet18
import logging  

data_dir = '/media/homes/areeba/thesis_projects/FYEMU/datasets/cifar10'
# Create a folder for logs if it doesn't exist
log_folder = '/media/homes/areeba/thesis_projects/FYEMU/logs'
os.makedirs(log_folder, exist_ok=True)
unique_id = int(time.time())
# Configure logging
log_file = os.path.join(log_folder, f'train_resnet18_{unique_id}.log')
logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def train_model_resnet18(epochs, max_lr, model, train_dl, valid_dl,grad_clip, weight_decay, opt_func):
    history = m.fit_one_cycle(epochs, max_lr, model, train_dl, valid_dl,
                                    grad_clip=grad_clip,
                                    weight_decay=weight_decay,
                                    opt_func=opt_func)

    torch.save(model.state_dict(), "ResNET18_CIFAR10_ALL_CLASSES.pt")
    model.load_state_dict(torch.load("ResNET18_CIFAR10_ALL_CLASSES.pt"))
    return model, history

if __name__ == '__main__':
    
    # run some code here
    transform_train = tt.Compose([
        tt.ToTensor(),
        tt.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = tt.Compose([
        tt.ToTensor(),
        tt.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    train_ds = ImageFolder(data_dir+'/train', transform_train)
    valid_ds = ImageFolder(data_dir+'/test', transform_test)


    batch_size = 256
    train_dl = DataLoader(train_ds, batch_size, shuffle=True, num_workers=3, pin_memory=True)
    valid_dl = DataLoader(valid_ds, batch_size*2, num_workers=3, pin_memory=True)


    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = resnet18(num_classes = 10).to(device = device)
    epochs = 40
    max_lr = 0.01
    grad_clip = 0.1
    weight_decay = 1e-4
    opt_func = torch.optim.Adam
    try:
        logging.info("\n************************ Model Training Started******************\n\n")
        model, history = train_model_resnet18(epochs=epochs, max_lr=max_lr, grad_clip=grad_clip, weight_decay=weight_decay, opt_func=opt_func, model=model, train_dl=train_dl, valid_dl=valid_dl)
        history = [m.evaluate(model, valid_dl)]
        logging.info(history)
        logging.info("\n********************* Model Training is Completed******************\n")
    except Exception as e:
        logging.error(f"Training failed because, {e}")