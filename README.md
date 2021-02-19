# Pegasus Workflow to Classify Cats and Dogs
![workflow](img/graph_tuned2.png)

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


![](img/catdog_single.png)


The workflow does hyperparameter optimiation on 3 different architectures and chooses best one.
Checkpoints are used to restart the training in case of time out or system failure.


![](img/cats_and_dogs.png)