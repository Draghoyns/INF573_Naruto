## Data
The dataset comes from the Naruto [Hand Sign Dataset](https://www.kaggle.com/datasets/vikranthkanumuru/naruto-hand-sign-dataset/data) on Kaggle.
I cleaned and augmented the data for more diversity.


## Usage
For using the program, it's very simple.

You simply have to run `main.py`.

By default, `main` will use a resnet model.

TODO : It is also possible to use YOLO, by uncommenting the last 2 lines in `main.py`

A window opens with the videostream from the webcam, and as the user makes signs, they will be detected in real-time.

The possible signs are the following :
![image](https://github.com/Draghoyns/INF573_HandSign/assets/145558291/76c82c9e-bbf2-469b-be87-c87c4deef25d)

That's it ! You just used your first justsu !

## TODO
- classifications using YOLO instead of ResNet
- or other models ?
- clean train data
- add custom data
- improvements ?
