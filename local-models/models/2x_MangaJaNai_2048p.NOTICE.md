# MangaJaNai model notice

- Original model: `2x_MangaJaNai_2048p_V1_ESRGAN_95k.pth`
- Source: https://github.com/the-database/MangaJaNai
- Release: https://github.com/the-database/MangaJaNai/releases/tag/1.0.0
- License: [Creative Commons Attribution-NonCommercial 4.0 International](https://creativecommons.org/licenses/by-nc/4.0/)

The model was converted from its original PyTorch checkpoint to a Core ML ML
Program with a fixed 1×3×256×256 input. The conversion uses Core ML's default
float16 compute precision. No training or fine-tuning was performed.
