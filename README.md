![alt text](https://www.dobitaobyte.com.br/wp-content/uploads/2016/10/keras0.png)
# Deep Learning Gender Detection model
![alt text](./illustrations/ezgif-3-cb71743555.gif)
## Presentation 

This is a Deep Learning gender detection model, a Convolutional Neural Network.
Built with Keras.

## Installation

I strongly adivse you to run this in a virtual environment ([Anaconda](https://www.anaconda.com/products/distribution), [Miniconda](https://docs.conda.io/en/latest/miniconda.html),[Miniforge](https://github.com/conda-forge/miniforge), [pyenv](https://github.com/pyenv/pyenv/blob/master/README.md), [venv](https://docs.python.org/fr/3/library/venv.html), etc...). Among all of these, i recommend you Anaconda, Miniconda or Miniforge, more suited for data-science utilisation.
Create a virtual environment with python 3.8.X.
If you are on MacOS Silicon like me, follow a [tutorial](https://www.youtube.com/watch?v=5DgWvU0p2bk) to install Tensorflow GPU if you can, or this [one](https://caffeinedev.medium.com/how-to-install-tensorflow-on-m1-mac-8e9b91d93706):


To install all the libraries/packages required, run this :

```bash
pip install -r requirements.txt
```
If there is any problem with the installation, install these one by one in your virtual environment if you have one :  
cvlib==0.2.7  
keras==2.9.0  
matplotlib==3.6.2  
numpy==1.22.3  
opencv_contrib_python==4.7.0.68  
pandas==1.5.2  
scikit_learn==1.2.0  
seaborn==0.12.2  
###### for tensorflow follow a tutorial depending on the machine you are working one
tensorflow==2.11.0  
tensorflow_macos==2.9.2  
argparse==1.1.0  
keras-tuner==1.1.3  



## Utilisation

The model is in the 'models' folder, which its hyperparameters were tuned with [keras tuner](https://keras.io/keras_tuner/), I achieved a val_accuracy of 96,8%.
You can directly open your camera and try it by running from the project folder :

```bash
python predict_gender.py
```
or if you want to detect the gender of people on images, run this :

```bash
python predict_gender.py --image_path <image path>
```
It will display the output on a cv2 window.
Notice that due to the detect_face function, it won't work well on too small images like the ones I used for training.
You might want to use full size images, with whole people, or images in which you can see the whole face.

Or if you want to have some fun, you can change the hyperparameters, change the model structure, which is defined in 'build.py'.
After doing all that you can train the model. It will save all the best models, choose which one you want to use.

## Model 
### Model Structure
Here is the model structure :
#### Using Keras plot_model
![alt text](./illustrations/model_architecture.png)
#### Using ann_vizualiser
![alt text](./illustrations/network.png)
### Model Training
The images were augmented, preprocessed using keras ImageDataGenerator and flow_from_directory.
The model was trained using Macbook Pro M1 2021 GPU, on a training dataset of 37608 images, and tested on a dataset of 11649 new images on only 15 epochs (you can improve the number of epochs if you want).
The dataset is mainly composed of white people faces, I am working on a much more balanced dataset.

### Model Evaluation
Here is the loss and accuracy obtained on the test data :

![alt text](./illustrations/test_score.png)

Here is the confusion matrix :

![alt text](./illustrations/confusion_matrix.png)

Here is the classification report on the new 11649 images :

![alt text](./illustrations/classification_report.png)

Here are some plots : 
![alt text](./illustrations/plots.png)
## Example
Detection by webcam
![alt text](./illustrations/example.png)
Press q or ESC to close the webcam window and stop the program.

Detection by image on several persons
![alt text](./illustrations/3people_detection_example.png)
Detection by image on one person
![alt text](./illustrations/female_detection_example.png)

#### Web App
![Detection by image on the web app](./illustrations/webapp_example.png)




