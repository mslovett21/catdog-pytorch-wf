#!/usr/bin/env python3

"""
MACHINE LEARNING WORKFLOWS - STEP  - HPO / Model Selection
Binary Classification of Images with Different Architectures

positional arguments:
  num_epochs       number of training epochs

optional arguments:
  -h, --help       show this help message and exit
  -arch arch_name  model architecture: basicnet | densenet121 | vgg16 (default: BasicNet)

Select one of the three models provided and choose number of epochs:
python train_model.py 7 10                  -  default architecture BasicNet
python train_model.py -arch vgg16 7 10      -  pretrained VGG16 with custom classifier part
python train_model.py -arch densenet121 7 10 -  pretrained DenseNet121 with custom classifier part


Details: we checkpoint Optuna study so that we can restart HPO in case of failure,
we also checkpoint model and training parameters so that we can restart Optuna study's trial
from the last checkpointed epoch of the trial.

Probably need to clean up some file in the TAR to not store too much infor about suboptimal 
trials in the study.


"""
import argparse
import os,sys
import tarfile
import joblib
import signal
import time

import torch
from torch import nn, optim
from skimage import io, transform
import torchvision.transforms as transforms

import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

from model_selection import BasicNet, PretrainedVGG16, PretrainedDenseNet121
from util_checkpoint import extract_checkpoints, load_checkpoint, checkpoints_tar,save_checkpoint
from data_loader import CatDogsDataset
from pytorch_tools import EarlyStopping



### -------------------------VARIABLES--------------------------------
DATASET_DIR = ""
DIR = os.getcwd()

CAT_TRAIN_PATH = DATASET_DIR + "train_Cat*.jpg"
DOG_TRAIN_PATH = DATASET_DIR + "train_Dog*.jpg"

CAT_VAL_PATH = DATASET_DIR + "val_Cat*.jpg"
DOG_VAL_PATH = DATASET_DIR + "val_Dog*.jpg"

# Tar with checkpoints
DEVICE = torch.device("cpu")#('cuda' if torch.cuda.is_available() else 'cpu')
MODEL = "basicnet"
EPOCHS = 10
CHECKPOINTS_FILES_LIST = []
STUDY = None
N_TRIALS = 10
BATCH_SIZE = 12
TAR_FILE = "basicnet_model.tar.gz"

### ------------------------- SIGTERM HANDLER  ----------------------
def sigterm_handler(signum, frame):
    print("SIGTERM recived")
    sys.exit(3)


### ------------------------- PARSER --------------------------------
arch_names = ["basicnet", "densenet121", "vgg16"]

parser = argparse.ArgumentParser(description='Binary Classification of Images with Different Architectures')
parser.add_argument('-arch', metavar='arch_name', type=str.lower, nargs=1, choices=arch_names, default="BasicNet",
	help='model architecture: ' +
	' | '.join(arch_names) +
	' (default: BasicNet)')	
parser.add_argument('epochs',  metavar='num_epochs', type=int, nargs=1, help = "number of training epochs")
parser.add_argument('trials',  metavar='num_trials', type=int, nargs=1, help = "number of HPO trials")


### -------------------------FOR DATALOADER --------------------------------
class ToTensorRescale(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, sample):
        image, label = sample["image"], sample["label"]
        image = image/255
        image = image.transpose((2, 0, 1))
        return {"image":torch.from_numpy(image),
                "label" :label}


def get_dataloaders():

    data_transforms  = transforms.Compose([ToTensorRescale()])
    train_dataset    = CatDogsDataset(CAT_TRAIN_PATH, DOG_TRAIN_PATH, transform=data_transforms)
    val_dataset      = CatDogsDataset(CAT_VAL_PATH, DOG_VAL_PATH, transform=data_transforms)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE,shuffle=True)
    val_dataloader   = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)

    return train_dataloader, val_dataloader


### -------------------------TRAIN AND VALIDATE LOOPS --------------------------------
def train(train_loader, model, criterion, optimizer):

    model.train()
    model.to(DEVICE)
    running_loss = 0

    for sample_batched in train_loader:        
        optimizer.zero_grad() 
        inputs, labels = sample_batched['image'].float(), sample_batched["label"]

        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)     
        log_ps = model(inputs)
        loss   = criterion(log_ps,labels)       
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    train_loss = running_loss/len(train_loader)

    return model, train_loss


def validate(val_loader, model, criterion):

    model.eval()
    model.to(DEVICE)
    accuracy = 0
    test_loss = 0

    with torch.no_grad():
        for sample_batched in val_loader:
            inputs,labels   = sample_batched["image"].float(), sample_batched["label"]
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            logs_ps = model(inputs)
            test_loss += criterion(logs_ps,labels)
            ps = torch.exp(logs_ps)
            top_ps, top_class = ps.topk(1,dim =1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor))
        test_loss_final = test_loss/len(val_loader)
        accuracy_final  = accuracy/len(val_loader)

    return model, test_loss_final, accuracy_final

### -------------------------HPO--------------------------------
def build_model(trial):

    model = None
    params_optim = None

    if MODEL == "vgg16":
        model = PretrainedVGG16().float()
        params_optim = model.parameters()
    elif MODEL == "densenet121":
        model = PretrainedDenseNet121().float()
        params_optim = model.parameters()
    else:
        p = trial.suggest_float("dropout", 0.2, 0.5)
        model = BasicNet(p)
        params_optim = model.parameters()

    return model, params_optim


def hpo_monitor(study, trial):
    joblib.dump(study,"hpo_study_checkpoint_" + MODEL + ".pkl")

def hpo_monitor_sigterm(final_prefix = ""):
    print("Inside final hpo_monitor")
    joblib.dump(STUDY,final_prefix + "hpo_study_checkpoint_" + MODEL + ".pkl")

def objective(trial):
    global CHECKPOINTS_FILES_LIST
    print("Performing trail {}".format(trial.number))

    # Generate the model.
    model, params_optim = build_model(trial)
    model.to(DEVICE)

    # Generate the optimizers.
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop"])
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    optimizer = getattr(optim, optimizer_name)(params_optim, lr=lr)
    early_stopping = EarlyStopping(patience=4, verbose=True)

    # Get the cat/dog dataset.
    train_loader, valid_loader = get_dataloaders()
    criterion = nn.NLLLoss()
    losses_dict= {'train': {}, 'test': {}, 'accuracy': {}}
    start_epoch = 1
    accuracy = 0

    for epoch in range(start_epoch,EPOCHS+1):
        model, train_loss = train(train_loader, model, criterion, optimizer)
        model, test_loss, test_accuracy = validate(valid_loader, model, criterion)
        current_metrics = [epoch, train_loss, test_loss, test_accuracy]
        losses_dict["train"][epoch], losses_dict["test"][epoch] = train_loss, test_loss
        losses_dict["accuracy"][epoch] = test_accuracy
        accuracy += test_accuracy
        time.sleep(5)

        if early_stopping.early_stop:
            break

        trial.report(test_accuracy, epoch)

        # Handle pruning based on the intermediate value.
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
            
    accuracy = accuracy/(EPOCHS)

    return accuracy



def create_study(hpo_checkpoint_file):
    global STUDY
    load_checkpoint_flag = True
    try:
        print("Trying to load an existing study...")
        STUDY = joblib.load("hpo_study_checkpoint_" + MODEL + ".pkl")
        print("read the study")
        todo_trials = N_TRIALS - len(STUDY.trials_dataframe())
        if todo_trials > 0 :
            print("There are {} trials to do out of {}".format(todo_trials, N_TRIALS))
            STUDY.optimize(objective, n_trials=todo_trials, timeout=600, callbacks=[hpo_monitor])
        else:
            pass
            #print("This study is finished. Nothing to do.")
    except Exception as e:
        print(e)
        print("New study")
        STUDY = optuna.create_study(direction = 'maximize', study_name = MODEL)
        STUDY.optimize(objective, n_trials=N_TRIALS, timeout=600, callbacks=[hpo_monitor])


### -------------------------MAIN--------------------------------
def main():

    global MODEL
    global EPOCHS
    global CHECKPOINTS_FILES_LIST
    global TAR_FILE
    global N_TRIALS
    print("Good start")
    print(DEVICE)

    try:
        signal.signal(signal.SIGTERM, sigterm_handler)
        
        args     = parser.parse_args()
        EPOCHS   = args.epochs[0]
        N_TRIALS = args.trials[0]

        if args.arch[0] == "densenet121":
            TAR_FILE = "densenet121_model.tar.gz"
            MODEL = "densenet121"
        elif args.arch[0] == "vgg16":
            TAR_FILE = "vgg16_model.tar.gz"
            MODEL = "vgg16"
        hpo_checkpoint_file = "hpo_study_checkpoint_" + MODEL + ".pkl"

        create_study(hpo_checkpoint_file)

    except Exception as e:
        print(e)
        pass
    finally:
        hpo_monitor_sigterm("final_")
  
    return 0



if __name__ == '__main__':
    main()




