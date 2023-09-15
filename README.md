
# Violence-Detector

This repository consists of a model trainer that detects violence.
- ❗️ __Python__, __pip__, and __Git__  must be installed on your device❗️
- Install python [Here](https://www.python.org/downloads/)
- Install pip [Here](https://pip.pypa.io/en/stable/installation)
- Install Git [Here](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git)
# How to use me!

Application/Model Deployment (Vertical Slice)
- Open up a terminal on your local device
- Clone repository on terminal using

  ` git clone https://github.com/darrenzwc/violence-detector.git `
- If this doesn't work, you can download the entire repository as a zip file instead.
- Change your current directory on the command line to the directory of where the files were downloaded.  `cd path/to/directory/violence-detector `

        - Ex: cd /notebooks/violence-detector 
- Type in ` ls ` in the terminal to view contents in the directory. It should at least have these following files: ` create-DS.py ` ` model.py ` ` video_dataset.py ` `  trainer.py ` ` requirements.txt`
- Type ` pip install -r requirements.txt` in the terminal to install all necessary libraries
- To run the demo, type `python app.py` in the terminal
- Momentarily, two links should pop up. Click on the public URL
- Close program using __CTRL +C__





~~DS Transformer~~ (console UI TBA for the final)

- ~~Download [Real Life Violence Situations Dataset](https://www.kaggle.com/datasets/mohamedmustafa/real-life-violence-situations-dataset)~~
- ~~Necessary files are in the __violence-detector__ directory~~
- ~~Install all needed libraries with `pip install -r requirements.txt`~~
- ~~❗️Make sure paths to `parent_dir` and `RawPath` are set accordingly in __create-DS.py__ ❗️~~
- ~~No need to change any paths if running through Paperspace Gradient~~
 - ~~Access and run immediately through [here](https://console.paperspace.com/darrenzwc/notebook/rc1kgwkifs50pj8)~~

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
| Dataloader | ✅|
| Model Outline #1| ✅|
| Model Outline #2| 🔨|
| Model Outline #3| ❌|
| Model Deployment website | ✅|
| Terminal UI | 🔨|
| Training| ✅|
 Results| ✅|


