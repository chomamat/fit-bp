# Script for interpolation of subsequent weather radar images

Weights for the model need to be downloaded here https://drive.google.com/open?id=1DD7e9MEq3JuqBpkhzRYm4nYQkNnrJF0Q first and saved in the folder, to use this script for interpolation.

Input images have to be grayscale with dimensions 96x96.

Script usage:
```python3 run.py --first examples/03_1.png --second examples/03_3.png --out examples/out.png```
