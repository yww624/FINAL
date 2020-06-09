# Transformer-Plus
Pytorch implementation of Transformer-Plus.  Main parts of the code are adapted from the implementation of [R-Transformer](https://arxiv.org/abs/1907.05572), [TCN](https://github.com/locuslab/TCN) and [Transformer](http://nlp.seas.harvard.edu/2018/04/03/attention.html). 


## Usage
Our repository is arranged as follows:
```
[Task Name] /
    data/ # contains the datasets
    experiment.py    # run ablation experiment ( model without Conv1d)
    experiment-2.py  # run ablation experiment ( model with Conv1d)
    model.py         # comtains the model (without Conv1d)
    model_native.py  # comtains the model
    utils.py # utility functions including dataset downloading and preprocessing
models /
    TransformerPlus.py                 # Transformer-Plus model without Conv1d
    TransformerPlusConv1d.py           # Transformer-Plus model 
```
The dataset for the "polyphonic music modeling" experiment is already included in audio/data/. 

For other experiments that are based on much larger datasets, the data needs to be downloaded (from [torchvision.datasets](https://pytorch.org/docs/stable/torchvision/datasets.html) or [observations](https://github.com/edwardlib/observations)) and then put into the "data" folder which should be created firstly.

When data is ready, the code can directly run with PyTorch  1.0.0.


