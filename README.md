# Wav2Letter+ attack based on IPC 
Code for ICLR_2020 paper (under review)

**Generating Robust Audio Adversarial Examples using Iterative Proportional Clipping** <br />
**[[Paper](https://openreview.net/forum?id=HJgFW6EKvH)]** <br />

### Installation
Install PyTorch if you haven't already. We currently implement our experiments on pytorch 1.1.0 with cuda9 and python3.7.

Install this fork for Warp-CTC bindings:

    $ git clone https://github.com/SeanNaren/warp-ctc.git
    $ cd warp-ctc
    $ mkdir build; cd build
    $ cmake ..
    $ make
    $ export CUDA_HOME="/usr/local/cuda"
    $ cd ../pytorch_binding
    $ python setup.py install

Note: If you encounter the problem of ***ModuleNotFoundError: No module named 'warpctc_pytorch._warp_ctc'***, you can copy the '~/anaconda2/envs/envs_name/lib/python3.7/site-packages/warpctc_pytorch-0.1-py3.7-linux-x86_64.egg/warpctc_pytorch' folder to the root directory of this project 'ICLR_2020_382/'.

    $ cp -r ~/warpctc_pytorch .

Install pytorch audio:

    $ sudo apt-get install sox libsox-dev libsox-fmt-all
    $ git clone https://github.com/pytorch/audio.git
    $ cd audio
    $ pip install cffi
    $ python setup.py install

Note: If you encounter the problem of ***error: command 'gcc' failed with exit status***, you can git clone https://github.com/pytorch/audio/tree/v0.2.0 (this branch can be download by clicking the **Download ZIP** button) before the installation.

If you want decoding to support beam search with an optional language model, install ctcdecode:

    $ git clone --recursive https://github.com/parlance/ctcdecode.git
    $ cd ctcdecode
    $ pip install .

Finally clone this repo and run this within the repo:

    $ pip install -r requirements.txt

### Inference 
We give two demos for researchers to reproduce our work.
1. Please download the [BaiDuYun](https://pan.baidu.com/s/1SuveHraSzv_Q9LhhOxxS4A) (code:ffgq).
2. Unzip the [4-gram.arpa.tar.gz] under the root directory, and put the [pretrained_wav2Letter.pth.tar] under the 'model_libri' folder.
3. 
Run the following code to attack from **original1** to **target1**:

    $ python attack_stage1.py --orig-path ./data/original/original1.wav --target-path ./data/target/target1.txt
    $ python attack_stage2.py --audio-path ./generated_stage1/original1_to_target1_stage1.wav --orig-path ./data/original/original1.wav --target-path ./data/target/target1.txt

Run the following code to attack from **original2** to **target2**:

    $ python attack_stage1.py --orig-path ./data/original/original2.wav --target-path ./data/target/target2.txt
    $ python attack_stage2.py --audio-path ./generated_stage1/original2_to_target2_stage1.wav --orig-path ./data/original/original2.wav --target-path ./data/target/target2.txt 
4. The generated adversarial audios are under the 'generated_stage2' folder.
