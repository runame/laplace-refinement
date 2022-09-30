# Posterior Refinement Improves Sample Efficiency in BNNs

This repository contains the code to run the experiments for the paper [Posterior Refinement Improves Sample Efficiency in Bayesian Neural Networks](https://arxiv.org/abs/2205.10041) (NeurIPS 2022), using our library [laplace](https://github.com/AlexImmer/Laplace/).

The files `al_train.py` and `ll_train.py` show how to refine an all-layer and last-layer Laplace posterior approximation _post hoc_. 
To run them, you can use the commands in the bash scripts in `run_scripts` with the corresponding name.
Specifically, the practically most relevant code for last-layer refinement is in [line 165-230](https://github.com/runame/laplace-refinement/blob/main/ll_train.py#L165-L230) in `ll_train.py`.

The method boils down to this very simple statement:
> Fine-tune your last-layer Laplace posterior with a normalizing flow. No need for a long, complicated flow; no need for many epochs for training the flow.

The file `uq.py` contains the code to run all the uncertainty quantification experiments on F-MNIST, CIFAR-10, and CIFAR-100. The commands to run the experiments are in the `run_*_uq.sh` files in the `run_scripts` folder.

_Note: Depending on your setup, you might have to copy the bash files in `run_scripts` to the root of this repo to be able to run the commands unchanged._

Please cite the paper if you want to refer to the method:
```bibtex
@inproceedings{kristiadi2022refinement,
  title={Posterior Refinement Improves Sample Efficiency in {B}ayesian Neural Networks},
  author={Agustinus Kristiadi and Runa Eschenhagen and Philipp Hennig},
  booktitle={{N}eur{IPS}},
  year={2022}
}
```
