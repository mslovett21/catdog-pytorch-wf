"""
MACHINE LEARNING WORKFLOWS - STEP  - Model Selection/ HPO

Binary Classification of Images with Different Architectures

usage: train_model.py [-h] best_model_hp num_epochs

positional arguments:
  best_model_hp  File with best HP for training
  num_epochs     Number of training epochs

optional arguments:
  -h, --help     show this help message and exit

python train_model.py hpo_study_checkpoint_densenet121.pkl 7  

"""


import argparse
import os
import tarfile
import matplotlib.pyplot as plt
import time
import signal

import torch
from torch import nn, optim
from skimage import io, transform
import torchvision.transforms as transforms
from pytorchtools import EarlyStopping

from util_checkpoint import extract_checkpoints, checkpoints_tar
from model_selection import BasicNet, PretrainedVGG16, PretrainedDenseNet121
from data_loader import CatDogsDataset


from IPython import embed

### -------------------------VARIABLES--------------------------------
DATASET_DIR    = "dev_data/"
CAT_TRAIN_PATH = DATASET_DIR + "train_Cat*.jpg"
DOG_TRAIN_PATH = DATASET_DIR + "train_Dog*.jpg"
CAT_VAL_PATH   = DATASET_DIR + "val_Cat*.jpg"
DOG_VAL_PATH   = DATASET_DIR + "val_Dog*.jpg"

# Tar with checkpoints
LOAD_CHECKPOINT = True
PRETRAINED_MODELS = ["densenet121", "vgg16"]
BATCH_SIZE = 12
ARCH_NAME = "basicnet"

### ------------------------- PARSER --------------------------------
parser = argparse.ArgumentParser(description='Binary Classification of Images with Different Architectures')
parser.add_argument('hp',  metavar='best_model_hp', type=str, nargs=1, help = "File with best HP for training")
parser.add_argument('epochs',  metavar='num_epochs', type=int, nargs=1, help = "Number of training epochs")






### -------------------------PLOT TRAIN/TEST LOSSES --------------------
def plot_losses(losses_dict, epoch,args ):
    train_lists = sorted(losses_dict["train"].items())
    test_lists = sorted(losses_dict["test"].items())
    accuracy_lists = sorted(losses_dict["accuracy"].items())
    
    plt.xlim([0,epoch])

    
    x1,y1 =zip(*train_lists)
    x2,y2 =zip(*test_lists)
    x3,y3 =zip(*accuracy_lists)

    plt.plot(x1,y1, label='Training loss')
    plt.plot(x2,y2, label='Validation loss')
    plt.plot(x3,y3, label='Validation accuracy')
    
    plt.legend(frameon=False)
    save_file_name = "./plots/loss_" +str(args.arch[0])+"_" + str(epoch)
    plt.savefig('{}.png'.format(save_file_name))
    plt.close()



### -------------------------CHECKPOINTS--------------------------------


def save_checkpoint(model,optimizer,current_metrics, checkpoints_files, losses_dict):
    
    filename='checkpoint_' + str(current_metrics[0]) +'.pth'
    checkpoint = { 'curr_epoch': current_metrics[0],
                   'model_state_dict' : model.state_dict(),
                   'optimizer_state_dict': optimizer.state_dict(),
                   'checkpoints_list': checkpoints_files,
                   'train_loss': current_metrics[1],
                   'test_loss': current_metrics[2],
                   'test_accuracy': current_metrics[3],
                   'losses_dict': losses_dict
    }

    checkpoints_files.append(filename)
    torch.save(checkpoint, filename)
    return checkpoints_files


def load_checkpoint(model,optimizer,filepath):
    
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['curr_epoch'] + 1
    losses_dict = checkpoint['losses_dict'] 

    return model, start_epoch, optimizer, losses_dict





### -------------------------FOR DATALOADER --------------------------------
class ToTensorRescale(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, sample):
        image, label = sample["image"], sample["label"]
        image = image/255
        image = image.transpose((2, 0, 1))
        return {"image":torch.from_numpy(image),
                "label" :label}



### -------------------------TRAIN AND VALIDATE LOOPS --------------------------------
def train(train_loader, model, criterion, optimizer, epoch, device):

    model.train()
    model.to(device)
    running_loss = 0

    for sample_batched in train_loader:
        
        optimizer.zero_grad() 
        inputs, labels = sample_batched['image'].float(), sample_batched["label"]

        inputs, labels = inputs.to(device), labels.to(device)     
        log_ps = model(inputs)
        loss   = criterion(log_ps,labels)
        
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    train_loss = running_loss/len(train_loader)
    print("Train Loss: {}".format(train_loss))

    return model, train_loss



def validate(val_loader, model, criterion,device):

    model.eval()
    model.to(device)
    accuracy = 0
    test_loss = 0

    with torch.no_grad():
        for sample_batched in val_loader:
            inputs,labels   = sample_batched["image"].float(), sample_batched["label"]
            inputs, labels = inputs.to(device), labels.to(device)
            logs_ps = model(inputs)
            test_loss += criterion(logs_ps,labels)
            ps = torch.exp(logs_ps)
            top_ps, top_class = ps.topk(1,dim =1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor))
        test_loss_final = test_loss/len(val_loader)
        accuracy_final  = accuracy/len(val_loader)
    print("Test Loss: {}".format(test_loss_final))
    print("Test Accuracy: {}".format(accuracy/len(val_loader)))

    return model, test_loss_final, accuracy_final


def build_model(model_name,best_model_spec):
    
    if model_name == "vgg16":
        model = PretrainedVGG16().float()
        params_optim = model.parameters()
    elif model_name == "densenet121":
        model = PretrainedDenseNet121.float()
        params_optim = model.parameters()
    else:
        model = BasicNet(best_model_spec["dropout_rate"]).float()
        params_optim = model.parameters()
    
    lr = best_model_spec["optimizer_lr"]
    
    if best_model_spec["optimizer_name"] == "Adam":
        optimizer = optim.Adam(params_optim, lr=lr)
    elif best_model_spec["optimizer_name"] == "RMSprop":
        optimizer = optim.RMSprop(params_optim, lr=lr)
    else:
        optimizer = optim.Adam(params_optim, lr=lr)
    return model, optimizer


### --------------------TRAIN MODEL--------------------------------
def main_worker(epochs,best_model_spec, checkpoints_files,args,device):
    global ARCH_NAME


    ARCH_NAME = best_model_spec["model_name"]

    data_transforms = transforms.Compose([ToTensorRescale()])

    train_dataset    = CatDogsDataset(CAT_TRAIN_PATH, DOG_TRAIN_PATH, transform=data_transforms)
    val_dataset      = CatDogsDataset(CAT_VAL_PATH, DOG_VAL_PATH, transform=data_transforms)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE,shuffle=True)
    val_dataloader   = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)


    model_name = best_model_spec["model_name"]
    start_epoch = 0
    model, optimizer = build_model(model_name,best_model_spec)
    model.to(device)
    losses_dict= {'train': {}, 'test': {}, 'accuracy': {}}
    criterion = nn.NLLLoss()

    if LOAD_CHECKPOINT:
        model, start_epoch, optimizer, losses_dict = load_checkpoint(model,optimizer,checkpoints_files[-1])

    early_stopping = EarlyStopping(patience=4, verbose=True)

    for e in range(start_epoch,epochs):
        print("{} out of {}".format(e+1, epochs))
        time.sleep(1)
        model, train_loss = train(train_dataloader, model, criterion, optimizer, epochs,device)
        model, test_loss, test_accuracy = validate(val_dataloader, model, criterion,device)
        current_metrics = [e,train_loss, test_loss,test_accuracy]
        losses_dict["train"][e] = train_loss
        losses_dict["test"][e] = test_loss
        losses_dict["accuracy"][e] = test_accuracy
        if early_stopping.early_stop:
            break
        if e % 2 == 0:
            checkpoints_files = save_checkpoint(model,optimizer, current_metrics, checkpoints_files, losses_dict)
    
    return checkpoints_files




def return_model_spec(hpo_checkpoint_file):

    f = open(hpo_checkpoint_file)
    params = eval(f.read())
    best_model = {}
    best_model["model_name"] = params["model_name"]
    parameters = params["parameters"]
    best_model["optimizer_name"] = parameters["optimizer"]
    best_model["optimizer_lr"]   = parameters["lr"]
    best_model["dropout_rate"]   = -1


    if  best_model["model_name"] == "basicnet":
        best_model["dropout_rate"] = parameters["dropout"]
        best_model["tar_file"] = "basicnet_model.tar.gz"
    elif best_model["model_name"] == "vgg16":
        best_model["tar_file"] = "vgg16_model.tar.gz"
    elif best_model["model_name"] == "densenet121":
        best_model["tar_file"] = "vgg16_model.tar.gz"

    return best_model




def main():
    global LOAD_CHECKPOINT

    checkpoints_files = []
    best_model_spec = {}
    best_model_spec["tar_file"] = "basicnet_model.tar.gz"
    
    def sigterm_handler(signum, frame):
        print("SIGTERM recived")
        print(checkpoints_files)
        checkpoints_tar(checkpoints_files,best_model_spec["tar_file"]) 
        sys.exit(1)
    
    try:
        args   = parser.parse_args()
        signal.signal(signal.SIGTERM, sigterm_handler)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        best_model_spec = return_model_spec(args.hp[0])
        best_model_spec["epochs"] = args.epochs[0]

        checkpoints_files = extract_checkpoints(best_model_spec["tar_file"], ".")
        
        if len(checkpoints_files) == 0:
            LOAD_CHECKPOINT = False

        checkpoints_files = main_worker(args.epochs[0],best_model_spec, checkpoints_files, args,device)
    except Exception as e:
        print(e)
    finally:
        checkpoints_tar(checkpoints_files,best_model_spec["tar_file"]) 
  
    return 0



if __name__ == '__main__':
    main()




