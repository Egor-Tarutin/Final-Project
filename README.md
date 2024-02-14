# face-editing.PyTorch

### Contents
- [Installation](#Installation)
- [Training](#training)
- [DjangoApp](#DjangoApp)
- [Examples](#Examples)
- [References](#references)

## Installation

- Python 3.9.0
- PyTorch 1.13
- Others can be installed using following command:
```Shell
pip install -r requirements.txt
```

## Training 
Model from https://github.com/zllrunning/face-parsing.PyTorch

1. Prepare training data:

    -- download [CelebAMask-HQ dataset](https://github.com/switchablenorms/CelebAMask-HQ)

2. 
	--  change file path in the `prepropess_data.py`  and run
```Shell
cd face_editing/image_segmentation/segmentation_model_rep
python prepropess_data.py
```

3. Train the model using CelebAMask-HQ dataset:
Just run the train script: 
```Shell
python train.py
```
If you do not wish to train the model, you can download [pre-trained model](https://drive.google.com/open?id=154JgKpzCPW82qINcVieuPH3fZ2e0P812) and save it in `res/cp`.


## DjangoApp

Change directory to django project directory:
```Shell
cd face_editing
```
And run command:
```Shell
python managa.py runserver
```

## References
- [BiSeNet](https://github.com/CoinCheung/BiSeNet)
- [face-parsing.PyTorch](https://github.com/zllrunning/face-parsing.PyTorch)
- [StableDiffusionInpainting](https://huggingface.co/runwayml/stable-diffusion-inpainting)
