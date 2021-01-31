<p  align="center"><img src="https://user-images.githubusercontent.com/42150335/106396946-5ae33b00-644e-11eb-9c00-b7f9127b4479.png" height=100>
  
<p  align="center">PyTorch implementation of RNN-Transducer

***

<p  align="center"> 
     <a href="https://github.com/sooftware/jasper/blob/main/LICENSE">
          <img src="http://img.shields.io/badge/license-Apache--2.0-informational"> 
     </a>
     <a href="https://github.com/pytorch/pytorch">
          <img src="http://img.shields.io/badge/framework-PyTorch-informational"> 
     </a>
     <a href="https://www.python.org/dev/peps/pep-0008/">
          <img src="http://img.shields.io/badge/codestyle-PEP--8-informational"> 
     </a>
     <a href="https://github.com/sooftware/conformer">
          <img src="http://img.shields.io/badge/build-passing-success"> 
     </a>
    
  RNN-Transducer are a form of sequence-to-sequence models that do not employ attention mechanisms. Unlike most sequence-to-sequence models, which typically need to process the entire input sequence (the waveform in our case) to produce an output (the sentence), the RNN-T continuously processes input samples and streams output symbols, a property that is welcome for speech dictation.   
    
![img](https://www.researchgate.net/publication/335044103/figure/fig3/AS:789632253952000@1565274408664/Recurrent-neural-network-RNN-transducer-structure-38.png)
  
## Installation
This project recommends Python 3.7 or higher.
We recommend creating a new virtual environment for this project (using virtual env or conda).
  
### Prerequisites
* Numpy: `pip install numpy` (Refer [here](https://github.com/numpy/numpy) for problem installing Numpy).
* Pytorch: Refer to [PyTorch website](http://pytorch.org/) to install the version w.r.t. your environment.  
* warprnnt_pytorch: Refer to [warp-transducer](https://github.com/HawkAaron/warp-transducer) to install warprnnt_pytorch.
  
## Usage

```python
import torch
import torch.nn as nn
from rnnt import RNNTransducer

batch_size, sequence_length, dim = 3, 12345, 80

cuda = torch.cuda.is_available()  
device = torch.device('cuda' if cuda else 'cpu')

inputs = torch.rand(batch_size, sequence_length, dim).to(device)
input_lengths = torch.IntTensor([12345, 12300, 12000])
targets = torch.LongTensor([[1, 3, 3, 3, 3, 3, 4, 5, 6, 2],
                            [1, 3, 3, 3, 3, 3, 4, 5, 2, 0],
                            [1, 3, 3, 3, 3, 3, 4, 2, 0, 0]]).to(device)
target_lengths = torch.LongTensor([9, 8, 7])

model = nn.DataParallel(RNNTransducer(num_classes=10)).to(device)

# Forward propagate
outputs = model(inputs, input_lengths, targets, target_lengths)

# Recognize input speech
outputs = model.module.recognize(inputs, input_lengths)
```
  
## Troubleshoots and Contributing
If you have any questions, bug reports, and feature requests, please [open an issue](https://github.com/sooftware/RNN-Transducer/issues) on github or   
contacts sh951011@gmail.com please.
  
I appreciate any kind of feedback or contribution.  Feel free to proceed with small issues like bug fixes, documentation improvement.  For major contributions and new features, please discuss with the collaborators in corresponding issues.  
  
## Code Style
I follow [PEP-8](https://www.python.org/dev/peps/pep-0008/) for code style. Especially the style of docstrings is important to generate documentation.  
  
## Reference
- [Sequence Transduction with Recurrent Neural Networks](https://arxiv.org/abs/1211.3711)
- [ZhengkunTian/rnn-transducer](https://github.com/ZhengkunTian/rnn-transducer)
  
## Author
  
* Soohwan Kim [@sooftware](https://github.com/sooftware)
* Contacts: sh951011@gmail.com
