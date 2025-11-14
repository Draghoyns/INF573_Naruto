## Data
The dataset comes from the Naruto [Hand Sign Dataset](https://www.kaggle.com/datasets/vikranthkanumuru/naruto-hand-sign-dataset/data) on Kaggle.
I cleaned and augmented the data for more diversity.


## Usage
For using the program, it's very simple.

The demo using ResNet is in `demo.py`, which you simply have to run to test.

The initial program using YOLO is in `simple_demo_without_post.py`, which also simply should be run to test it.

In both cases, a window opens with the videostream from the webcam, and as the user makes signs, they will be detected in real-time.

The possible signs are the following :
![image](https://github.com/Draghoyns/INF573_HandSign/assets/145558291/76c82c9e-bbf2-469b-be87-c87c4deef25d)

That's it ! You just used your first justsu !

## TODO
- classifications using YOLO instead of ResNet
- clean train data
- add custom data
- improvements ?
