# Detect License Plates And Blur It

This project is for detect car license plate and blur it. It was made for my work/study, not a big professional project but it work inchallah.

## What is inside

- `analytique/` first try with OpenCV, canny and contours
- `CNN/final/` the final app and model files
- `CNN/test/` some test scripts and examples
- `CNN/training files/` dataset and files for training

## Libraries

You need Python and this libraries:

```bash
pip install ultralytics opencv-python numpy customtkinter Pillow PyYAML imutils matplotlib
```

Maybe use virtual env if you want, it is better.

## How to run

Go to final folder:

```bash
cd CNN/final
python main.py
```

For testing images or video you can check the files in `CNN/test`.

## Training

The dataset is in:

```text
CNN/training files/dataset
```

There is also annotation file and training script. If you want train again you can run the training script, but first check paths because maybe they are different in your pc.

## Note

The big original zip file is not pushed because GitHub doesn't accept file more than 100 MB without LFS. The extracted files are here in the project.
