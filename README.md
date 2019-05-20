## CRFs as Mask Refinement for Binary Segmentation
This is the code for project of Machine Learning: Data to Models. A description of the project can be found at: https://www.overleaf.com/read/vqnfkjpvkpfj

1. Prepare Dataset to train FCNs:
Download DAVIS-2017-trainval-Full-Resolution from https://davischallenge.org/davis2017/code.html
```
python list_train_val.py
python crop.py
```
2. Train FCN to get unary potentials:
* Install torchfcn: follow instructions here:https://github.com/wkentaro/pytorch-fcn
* Replace the original `torchfc/datasets/voc.py` with `voc.py`, `torchfc/__init__.py with __init__.py` and `examples/voc/train_fcn8s_atonce.py` with `train_fcn8s_atonce.py`. Copy `test_fcn8s_atonce.py` to `examples/voc/`.
* Copy data to the specified path. In my computer, it is `~/data/datasets/VOC/benchmark_RELEASE/dataset/`
* Start training: in `examples/voc` run
```
python train_fcn8s_atonce.py -g 0
```
* Getting Unary Potentials: in `examples/voc` run
```
python test_fcn8s_atonce.py -g 0
```
3. Prepare inputs for Setting 1
```
python choose_subset.py
python make_dataset.py
```
4. Run Setting 1:
* Install requirements:
```
pip install pystruct
pip install cvxopt
```
* Train:
```
python image_segmentation.py
```
* Evaluate: copy `test_pystruct.py` to `examples/voc`
```
python test_pystruct.py
```
5. Run Setting 2:
* Install ConvCRF: follow instructions here: https://github.com/MarvinTeichmann/ConvCRF
* Copy `convcrf` folder, `train_convcrf.py`, `test_convcrf.py` to `examples/voc`
* Training and Testing:
```
python train_convcrf -g 0
python test_convcrf -g 0
```
