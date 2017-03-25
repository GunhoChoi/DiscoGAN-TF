# DiscoGAN in Tensorflow

Tensorflow implementation of [Learning to Discover Cross-Domain Relations with Generative Adversarial Networks](https://arxiv.org/abs/1703.05192)

<img src="./assets/discogan.png" width="60%">

## Requirements

- Tensorflow 1.0.1
- Python 3.5.2
- Pillow
- wget

## Download code
~~~~
git clone https://github.com/GunhoChoi/DiscoGAN_TF.git
cd DiscoGAN_TF
~~~~~

## Download Image
~~~
python3 down_resize_crop.py
~~~
## Train Model
~~~
python3 DiscoGAN.py
~~~
## Result

 under training..
 
## Insight

In order to get better results, the generator needs to be smarter than discriminator.
Also, greater latent size of generator is required in order to get better generated images.

### Links

   - Official PyTorch implementation (https://github.com/SKTBrain/DiscoGAN)
   - Carpedm20's PyTorch implementation (https://github.com/carpedm20/DiscoGAN-pytorch)
