# Model Conversion Scripts

Note: Python 3.9 must be used (probably, that's just what I had to use for things to work for me).

```sh
python3.9 -m venv env
source env/bin/activate
pip install torch==2.5.0 coremltools
```

## Waifu2x

```sh
# get models
wget https://github.com/nagadomi/nunif/releases/download/0.0.0/waifu2x_pretrained_models_20250502.zip
unzip waifu2x_pretrained_models_20250502.zip
# get dependencies
git clone git@github.com:nagadomi/nunif.git
pip install Pillow torchvision
# run conversion
python waifu2x_upconv7_convert.py pretrained_models/upconv_7
```

## Real-ESRGAN (and others, with spandrel)

Model files can be found [here](https://github.com/xinntao/Real-ESRGAN/blob/master/docs/model_zoo.md).

```sh
pip install spandrel
python spandrel_convert.py path/to/model.pth
```

The converter also supports ESRGAN models that use pixel unshuffle, including
MangaJaNai, when the fixed 256x256 input is divisible by the shuffle factor.

## Real-CUGAN (local experiment)

The experimental converter expects the upstream `upcunet_v3.py` source and a
Real-CUGAN 2x checkpoint. It emits a fixed 256x256-input Core ML package for
local testing in Aidoku:

```sh
python realcugan_convert.py \
  /path/to/up2x-latest-conservative.pth \
  --source-file /path/to/Real-CUGAN/upcunet_v3.py \
  --output RealCUGAN_up2x_conservative.mlpackage
```

The source and weights come from the upstream Real-CUGAN project. This branch
is for local evaluation only; verify model-weight redistribution terms before
publishing the generated package.
