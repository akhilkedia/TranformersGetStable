# Code for DSLM

## Simulation

### Environment Setup

Use `pip install -r simulations/requirements.txt`. Only requires torch, numpy, tqdm and matplotlib. A CUDA GPU is required as well, tested on 1 A100.

### Running

Run the file `run_all.sh` to reproduce all our simulations for transformer components. The file `approximations.py` can also plots the approximations of RELU/MLP covariance.

### Expected Output

Expected output is provided in `expected_output.txt`

## DSLM Signal Propagation Figures

These figures show Unit forward and back change in variances.

### Environment Setup

Use `conda env create -f environment.yml`. Tested on 8x A100 80GB. Same enviroment is also required for Xavier and Pre-training.

Also requires pre-training data from `bert_wiki_pretraining/prepare_data.sh`

### Running

cd into `DSLM_signal_propagation_figures` and run the file `make_figs.sh`.

It will make xavier figures `preln_forward.png`, `preln_backward.png` and `postln_backward.png` and exit.

These files are already included, delete these `.png` files to recreate.

### Expected Output

<img src="DSLM_signal_propagation_figures/preln_forward.png" alt="drawing" width="40%"/>
<img src="DSLM_signal_propagation_figures/Preln_backward.png" alt="drawing" width="40%"/>
<img src="DSLM_signal_propagation_figures/postln_backward.png" alt="drawing" width="40%"/>

## Xavier Signal Propagation Figures

These figures show forward and back change in variances for vanilla transformer models.

### Running

cd into `xavier_signal_propagation_figures` and run the file `make_figs.sh`.

It will make xavier figures `preln_forward.png`, `preln_backward.png` and `postln_backward.png` and exit.

These files are already included, delete these `.png` files to recreate.

### Expected Output

<img src="xavier_signal_propagation_figures/preln_forward.png" alt="drawing" width="40%"/>
<img src="xavier_signal_propagation_figures/preln_backward.png" alt="drawing" width="40%"/>
<img src="xavier_signal_propagation_figures/postln_backward.png" alt="drawing" width="40%"/>

## Baseline BERT Model Pretraining and Finetuning

### Running

1. cd into `bert_wiki_pretraining`
1. Run the file `prepare_data.sh` to download and process the pre-training dataset. This is Wikipedia from TFDS.
1. Run the file `run_bert_wiki.sh` to run the pretraining.
1. Run the files `examples/run_mnli.sh`, `examples/run_qqp.sh`, `examples/run_race.sh` to run finetuning.