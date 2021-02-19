# Pegasus Workflow to Classify Cats and Dogs

<img src="imgs/catdog-wf-bestmodel.png"  height="300"/>

### Usage
1. `git clone git@github.com:mslovett21/catdog-pytorch-wf.git`
2. `pip3 install requirements.txt`
    - This install all needed dependencies.
3. run notebook of your choice
4. `wf-final-tune.ipybn` is a work in progress. (See goal).

### Goal 

Fix the repo:
 - ensure the checkpointing works as required


# Pegasus Workflow to Classify Cats and Dogs
Detailed walk through steps of training a single architecture. Here VGG16.


![](imgs/catdog_single.png)


The workflow does hyperparameter optimiation on 3 different architectures and chooses best one.
Checkpoints are used to restart the training in case of time out or system failure.


![](imgs/cats_and_dogs.png)