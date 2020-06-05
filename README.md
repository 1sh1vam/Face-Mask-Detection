# Face Mask Detection

Implementation of Face Mask Detection to restrict people without wearing a mask from entering inside any mall or airport etc. We have used Tensorflow, Keras and opencv to implement this.

# Requirements

* Python3
* Tensorflow 2.0
* Keras
* Numpy
* cv2
* Matplotlib

# Insatallation Instructions
```
$ pip install tensorflow(or any other package name)
```
# Usage
If you want to train the model then change the path in "split_data" function of face_mask_detection.py/face_mask_detection.ipynb file.

To detect mask on your video change the path of cv.VideoCapture("Path of your Video") in detection.ipynb/detection.py. And then run python file to detect masks.

# Results

![result](https://github.com/1sh1vam/Face-Mask-Detection/blob/master/Data/result.gif)

# Improvement Scope
There is always a scope of improvement in any project. Few ways to improve this project are:
* Using "Temperature Sensor" to restrict people whose body temperature is above normal body temperature e.g 100.
* Use "Distance Measuring Sensors" to ensure social distancing.
If you think there can be more ways to improve this system please write to me(https://www.linkedin.com/in/1sh1vam/)


# Reference
* [DataSet](https://github.com/prajnasb/observations/tree/master/experiements/data)

* [Original Video that I used to show result](https://www.youtube.com/watch?v=b1Y3FSAxj3g)
