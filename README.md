# FastDiffFaceAttack
Code for thesis "Personalized Adversarial Perturbations for Facial Privacy Protection Using Latent Diffusion Models."

# Setup
1. Create a virtual encironment with python 3.8.
```
conda create -n env_name python=3.8
```
2. To install dependencies, run the following commands in the root.
```
pip install torch==1.12.0+cu113 torchvision==0.13.0+cu113 torchaudio==0.12.0 --extra-index-url https://download.pytorch.org/whl/cu113
pip install -r requirements.txt
```
3. To construct the dataset, download the [CelebAMask-HQ](https://github.com/switchablenorms/CelebAMask-HQ) dataset from their official page and place the .zip file in the project root directory, then run the following commands.
```
unzip CelebAMask-HQ.zip
python generate_dataset.py
```

