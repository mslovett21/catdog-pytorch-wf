# Pegasus Workflow to Classify Cats and Dogs

### Usage
1. `git clone git@github.com:mslovett21/catdog-pytorch-wf.git`
2. `pip3 install requirements.txt`
    - This install all needed dependencies.
3. run notebook of your choice
4. `wf-final.ipybn` is a work in progress. (See goal).

### Goal 

Fix the repo:
 - ensure the checkpointing works as required
 - get rid of data split
 - allow pre-precessing in parallel
 - add first job that appends prefix of name of class to file
 - improve utils


# Pegasus Workflow to Classify Cats and Dogs
Detailed walk through steps of training a single architecture. Here VGG16.


![](img/catdog_single.png)


The workflow does hyperparameter optimiation on 3 different architectures and chooses best one.
Checkpoints are used to restart the training in case of time out or system failure.


![](img/cats_and_dogs.png)