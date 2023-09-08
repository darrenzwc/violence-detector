
# Violence-Detector

This repository consists of a model trainer that detects violence.

# How to use me!

DS Transformer
- Download [Real Life Violence Situations Dataset](https://www.kaggle.com/datasets/mohamedmustafa/real-life-violence-situations-dataset)
- Necessary files are in the __violence-detector__ directory
- Install all needed libraries with `pip install -r requirements.txt`
- ❗️Make sure paths to `parent_dir` and `RawPath` are set accordingly in __create-DS.py__ ❗️
- No need to change any paths if running through Paperspace Gradient
 - Access and run immediately through [here](https://console.paperspace.com/darrenzwc/notebook/rc1kgwkifs50pj8)

# Visual representation of file and directory restructure

### Real Life Violence Situations Dataset
```
violence_ds
│
├───Nonviolence
│       ├───NV_1.mp4
│       .
|       └───NV_411.mp4
│       
│
└───Violence
        ├───V_1.mp4
        .
        └───V_431.mp4
 
```
### VD-DS
```
VD-DS
│
├───annotations.txt
├───NonViolent
│       ├───0001_NV_1
│       │     ├───img_00001.jpg
│       │     .
│       │     └───img_00017.jpg
│       └───0002_NV_192
│             ├───img_00001.jpg
│             .
│             └───img_00018.jpg
│
└───Violent
        ├───0001_V_41
        │     ├───img_00001.jpg
        │     .
        │     └───img_00015.jpg
        └───0002_V_192
              ├───img_00001.jpg
              .
              └───img_00015.jpg
```
## Progress Reference

| Git Section             | Status                                                                    |
| ----------------- | ------------------------------------------------------------------ |
| DS Transformer |✅|
| Dataloader | 🔨|
| Model Outline #1| ✅|
| Model Outline #2| 🔨|
| Model Outline #3| ❌|
| Training| ❌|
 Results| ❌|

