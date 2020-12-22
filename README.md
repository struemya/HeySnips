# HeySnips ML on MCU 2020
## train.py:

Train model using:
```
python train.py --flag1 <value1> --flag2 <value2>
```
### Flags:
- data_root: Root directory of the data
- exp_root: Root directory 
- batch_size: Batch size used during training
- epochs: Maximum number of epochs 
- lr: Learning rate
- patience: Patience setting for the learning rate scheduler
- num_feat: Number of input features. A precomputed dataset for this number of features is required.
- dilation_stages: Number of Layers with increasing dilations. Maximum dilation will be 2**dilatation_stages
- num_stacks: Number of Stacks, each stack consists uf Residual Blocks of 1D Convolutions
- num_filter: Number of convolutional filters
- bn: Whether to use Batch Normalization
- use_skip_connections: Whether to use skip connections
- return_sequences: Whether to return sequences (for sequence to sequence prediction) or the last element in the output sequence (for single label prediction)
- debug: Debug Mode, use Eager Execution

## evaluate.py:
Evaluate Model performance using:
```
python evaluate.py --exp_root "path to experiments root" --exp_name "name of the experiment folder" --exp_nr "number of the experiment"
```
Creates pickle file "metrics.p" in the test folder of the experiment.

## evaluate_quant.py:
Evaluate quantized model performance using:
```
python evaluate_quant.py --exp_root "path to experiments root" --exp_name "name of the experiment folder" --exp_nr "number of the experiment"
```
The model will be quantized and evaluated. Creates pickle file "quant_metrics.p" in the test folder of the experiment.
## quantize.py:
Quantize using:
```
python quantize.py --exp_root "path to experiments root" --exp_name "name of the experiment folder" --exp_nr "number of the experiment"
```
Creates the int8 quantized .tflite model in the experiment folder.
