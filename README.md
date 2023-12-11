# EA_RePOSE

 

## Prerequisites
- Python >= 3.6
- Pytorch == 1.9.0
- Torchvision == 0.10.0
- CUDA == 10.1

## Downloads
The dataset parts are from the Downloads part in RePOSE(https://github.com/sh8/RePOSE.git)

Our training results w/o dqn can be downloaded from [this part](https://drive.google.com/drive/folders/1gVRkrTW8oyjWFri4_5ZlneEyO_ejti8X?usp=sharing)

     
    $ ROOT=/path/to/EA_RePOSE
    $ mkdir  $ROOT/bestresult
     
    
Then copy the training weights in it

Dqn results can be downloaded from [this part](https://drive.google.com/drive/folders/1gdcLg-kuycxDlAUlaeo89bKEPZ-NkXX1?usp=sharing)

    
    $ ROOT=/path/to/EA_RePOSE
    $ mkdir  $ROOT/bestresult_dqn
    
    
Then copy the training weights in it



## Installation

1. Set up the python environment:
    ```
    $ pip install torch==1.9.0 torchvision==0.10.0
    $ pip install Cython==0.29.17
    $ sudo apt-get install libglfw3-dev libglfw3
    $ pip install -r requirements.txt

    # Install Differentiable Renderer
    $ cd renderer
    $ python3 setup.py install
    ```
2. Compile cuda extensions under `lib/csrc`:
    ```
    ROOT=/path/to/EA_RePOSE
    cd $ROOT/lib/csrc
    export CUDA_HOME="/usr/local/cuda-10.1"
    cd ../camera_jacobian
    python setup.py build_ext --inplace
    cd ../nn
    python setup.py build_ext --inplace
    cd ../fps
    python setup.py
    ```
3. Set up datasets:
    ```
    $ ROOT=/path/to/EA_RePOSE
    $ cd $ROOT/data

    $ ln -s /path/to/linemod linemod
    $ ln -s /path/to/linemod_orig linemod_orig
 
    $ cd $ROOT/cache/LinemodTest
    $ unzip ape.zip benchvise.zip .... phone.zip
 
## Testing w/o DQN

### Evaluate the ADD(-S) score

1. Generate the annotation data:
    ```
    python run.py --type linemod cls_type ape model ape
    ```
2. Test (The method of initial poses can be modified in configs/linemod.yaml):
    ```
    # Test on the LineMOD dataset with results of PVNET
    $ python run.py --type evaluate --cfg_file configs/linemod.yaml cls_type ape model ape mode PVNET
    
    # Test on the LineMOD dataset with results of PoseCNN
    $ python run.py --type evaluate --cfg_file configs/linemod.yaml cls_type ape model ape mode PoseCNN
 
    ``` 
3. TensorRT version:

Please convert weight file('.pth') into tensorRT file according to [TensorRT](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html).

Then change branch into main_tensorrt:
 ```
 git checkout main_tensorrt
 ```
Then run the above orders again.


## Testing with DQN

### Evaluate the ADD(-S) score
1. Change branch into main_dqn:
 ```
 git checkout main_dqn
 ```
 
2. Test (The method of initial poses can be modified in configs/linemod.yaml):
    ```
    # Test on the LineMOD dataset with results of PVNET
    $ python run.py --type dqn --cfg_file configs/linemod.yaml cls_type ape model ape mode PVNET
    
    # Test on the LineMOD dataset with results of PoseCNN
    $ python run.py --type dqn --cfg_file configs/linemod.yaml cls_type ape model ape mode PoseCNN
 
    ``` 
3. TensorRT version:

Then change branch into main_dqn_tensorrt:
 ```
 git checkout main_dqn_tensorrt
 ```
Then run the above orders again.



    
## Acknowledgement
Our code is largely based on [Repose](https://github.com/sh8/RePOSE.git) and [PVNET](https://github.com/zju3dv/pvnet.git).  Thanks for their sharing.

