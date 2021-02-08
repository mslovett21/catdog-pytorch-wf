import tarfile
import torch
import joblib

PRETRAINED_MODELS = {"densenet121":26 , "vgg16": 30}

### -----------------------CHECKPOINTS --------------------------------

def save_checkpoint(study,model,optimizer,current_metrics, checkpoints_files, losses_dict,trial,model_name):
    model_state_dict = {}

    if model_name in PRETRAINED_MODELS.keys():
        whole_state_dict = model.state_dict()
        all_keys = list(whole_state_dict.keys())
        classifier_keys = all_keys[PRETRAINED_MODELS[model_name]:]

        for k,v in whole_state_dict.items():
            if k in classifier_keys:
                model_state_dict[k] = v
    else:
        model_state_dict = model.state_dict()

    filename='chckpnt_trial_' + str(trial.number) +'_epoch_'+ str(current_metrics[0])+"_"+ model_name +'.pth'
    checkpoint = { 'trail': trial.number,
                 'run_parameters': str(trial.params),
                 'curr_epoch': current_metrics[0],
                 'model_state_dict' : model_state_dict,
                 'optimizer_state_dict': optimizer.state_dict(),
                 'checkpoints_list': checkpoints_files,
                 'train_loss': current_metrics[1],
                 'test_loss': current_metrics[2],
                 'test_accuracy': current_metrics[3],
                 'losses_dict': losses_dict,
                 'optuna_study':study
    }

    checkpoints_files.append(filename)
    torch.save(checkpoint, filename)
    return checkpoints_files


def extract_checkpoints(path, output_directory):
    print("Try to extract checkpoint from TAR")
    try:
      tar = tarfile.open(path)
      tar.extractall(path=output_directory)
      return tar.getnames()
    except (tarfile.ReadError, IOError):
      return []


def load_checkpoint(model,optimizer,filepath, model_name):
    print("Try to load checkpoint")

    checkpoint = torch.load(filepath)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['curr_epoch'] + 1
    losses_dict = checkpoint['losses_dict']
    if mode_name in PRETRAINED_MODELS:
        model.classifier.load_state_dict('model_state_dict')
    else:
        model.load_state_dict(checkpoint['model_state_dict'])

    return model, start_epoch, optimizer, losses_dict


def checkpoints_tar(checkpoints_files, tar_file):
    tar = tarfile.open(tar_file,"w:gz")
    for name in checkpoints_files:
      tar.add(name)
    tar.close()



