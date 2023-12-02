![Music](https://i.chzbgr.com/full/7701429248/h9E7A7628/hes-a-fan)

# _**Music Genre Identification**_
How many times we have wondered, what might have been a songs genre, sometimes it might be easy and sometimes it might make our brains go wild, so this is it, this program allows us to predict the genre of the music using Transfer learning. The flow is as follows first we load the music file, split the songs into chunks, then load the model using load model (), using the loaded model, predict on the music which you ahve loaded earlier and since the output will be in the form of array, we use argmax() to extract the highest probability of the music genre the model has predicted.

# _**Base Paper**_
+ https://www.researchgate.net/publication/324218667_Music_Genre_Classification_using_Machine_Learning_Techniques
+ https://www.researchgate.net/publication/329396097_Music_Genre_Classification_and_Recommendation_by_Using_Machine_Learning_Techniques

# _**Algorithm Description**_

# **Transfer Learning:**
Transfer Learning is a method where we train our dataset with a model which was already trained on such type of problem that we are working on i.e. Since we are dealing with a classification problem i.e., classifying which genre the music belongs to so we use a specific transfer learning algorithm which is made of classification problem. So, the model we are using in this project is VGG16. VGG16 is basically a 16-layer convolutional neural network which was trained on image net datset which consists of 14 million images, with around 1000 classes. VGG16 is most preferred classification model if we wanted to use transfer learning in our project. This model along with such good accuracy and performance, it also comes with some disadvantages such as exploding gradient descent problem due to more than 100 million parameters for training and due to the size of the model i.e., 528 MB,although it might be a very small size but this will take a lot of time to run and even download in many systems.

![TF](https://www.mdpi.com/sensors/sensors-23-00570/article_deploy/html/images/sensors-23-00570-g001.png)

**References:**
+ https://machinelearningmastery.com/transfer-learning-for-deep-learning/

# _**How to Execute?**_
So, before execution we have some pre-requisites that we need to download or install i.e., anaconda environment, python and a code editor.
**Anaconda**: Anaconda is like a package of libraries and offers a great deal of information which allows a data engineer to create multiple environments and install required libraries easy and neat.

**Download link:**

![Anaconda](https://1.bp.blogspot.com/-UJ1Ws2zZ9V4/TtMbG2ynJiI/AAAAAAAABbM/m6t2kuEhKdY/s1600/The-biggest-anaconda-snake-3.jpg)

https://www.anaconda.com/

**Python**: Python is a most popular interpreter programming language, which is used in almost every field. Its syntax is very similar to English language and even children and learning it nowadays, due to its readability and easy syntax and large community of users to help you whenever you face any issues.

**Download link:**

![Python](https://i0.wp.com/reptileworldfacts.com/wp-content/uploads/2019/05/male-blonde-super-tiger-reticulated-python.jpg?resize=351%2C351&ssl=1)

https://www.python.org/downloads/

**Code editor**: Code editor is like a notepad for a programming language which allows user to write, run and execute program which we have written. Along with these some code editors also allows us to debug, which usually allows users to execute the code line by line and allows them to see where and how to solve the errors. But I personally feel visual code is very good to work with any programming language and makes a great deal of attachment with user.

**Download links:**

![Vs code](https://schwabencode.com/contents/logos/VS2019-Badge.png) ![Pycharm](https://i0.wp.com/scracked.com/wp-content/uploads/2020/01/PyCharm-2019.3.4-Crack.png?fit=200%2C200&ssl=1)

+ https://code.visualstudio.com/Download, 
+ https://www.jetbrains.com/pycharm/download/#section=windows

# _**How to create a new environment and configure jupyter notebook with it.**_
Let us define an environment and why we need different environments. An environment is a collection of libraries that are required to run our project. When we already have an environment with the necessary libraries, why do we need a new environment?
To avoid version mismatches, we create a new environment for each project. For example, in your previous project, you used "tf env" with tensorflow 2.4 and keras 2.4, but in your current project, you must use tensorflow 2.6 and keras 2.6. If you continue your project in the "tf env" environment, there will be a version mismatch and you will need to update tensorflow and keras, but this will cause problems with the previous project's execution. To avoid this, we create a new environment with tensorflow 2.6 and keras 2.6 and resume our project.

# _**How to create an environment in anaconda.**_
+ Type “conda create –n <<name_of_your_env>>”
example: conda create -n env
+ It will ask to proceed with the environment location, type ‘y’ and press enter.
+ When you press ‘y’, the environment will be created. To activate your environment type conda activate <<your_env_name>> . E.g., conda activate myenv.
+ You can see that the environment got changed after conda activate myenv line. It changed from “base” to “myenv” which means you are now working in “myenv” environment.
+ To install a library in your virtual environment type pip install <library_name>.
e.g., pip install pandas
+ Instead of installing libraries one by one you can even install by bunch, i.e., we have a txt file called requirements.tx which consists of all the libraries required to proceed with the project, so we can use it.
+ so, before installing requirements.txt, make sure you are in the specific path where your requirements.txt is located, basically this file is located in the folder where our executable files are located, so we need to move to that directory by following command.
**cd C:\folder_name**
+ Here A -> drive, folder name -> path where your executable file is saved
+ I go to that file path in anaconda using cd command 
1.	Go to drive where your project file is.
2.	Go to the path of your project using cd <path>
3.	Type pip install –r requirements.txt 
+ And all your required libraries will be downloaded and you can start your project.
+ But if you want to use jupyter notebook on the new environment you have to set it up for the new environment.
+ After you have installed all the libraries and created an environment, you need an editor to run the code, that is starting jupyter notebook, as soon as you enter jupyter notebook in the terminal you will definitely get this error. “Jupiter” is not recognized as an internal or external command.
So, to solve it it we have 2 commands.
1.	conda install –c conda-forge jupyterlab
2.	conda install –c anaconda python
Now you are ready to use jupyter on this environment and start with your project!

![thanks](https://media1.giphy.com/media/ZfK4cXKJTTay1Ava29/giphy.gif)
  
  
# _**Steps to execute**_
**Note:** Make sure you have added path while installing the software’s.(Supports Python==3.8 only)

1.	Install the prerequisites/software’s required to execute the code.
2.	Press windows key and type in anaconda prompt a terminal opens up.
3.	Before executing the code, we need to create a specific environment which allows us to install the required libraries necessary for our project.
•	Type conda create -name “env_name”, e.g.: conda create -name project_1
•	Type conda activate “env_name, e.g.: conda activate project_1
4.	Make sure you are in the correct path in your terminal, where you have saved your executable file/folder. E.g.: cd A:\project\AI\Completed\project_name, then press enter.
5.	Install necessary libraries from requirements.txt file provided.
6.	Run pip install -r requirements.txt or conda install requirements.txt (Requirements.txt is a text file consisting of all the necessary libraries required for executing this python file. If it gives any error while installing libraries, you might need to install them individually.)
7.  Download & Unzip The Datset.rar File from below link & Paste The File Location where ever It's required.
+ https://drive.google.com/file/d/1M-Ior5zb_OFf9XjYvMJNjGhopLsVIUt6/view?usp=sharing
8.  Click on classification_cnn_vgg.ipynb to get the results.

# _**Data Description**_
The particular dataset was downloaded from Kaggle data repository, which consists of 10 classes and each class consists of around more than 100+ audio files. There are 10 music genres which are included in this dataset such as Blues, Classical, Country, Disco, Hip-Hop, Jazz, Metal, Pop, Reggae and Rock. Each audio file has around 30 second timestamp. Below given link can be accessed to download the dataset from the Kaggle data repository.

![Dataset](https://miro.medium.com/v2/resize:fit:1200/1*c8wR6BtKoF-kG5lqqcI7hA.gif)

 # _**Issues Faced.**_
1. We might face an issue while installing specific libraries.
2. Make sure you have the latest version of python or 3.8, since sometimes it might cause version mismatch.
3. Adding path to environment variables in order to run python files and anaconda environment in code editor, specifically in visual studio code.

# _**Note:**_
**All the required data hasn't been provided over here. Please feel free to contact me for any issues. You can also download the dataset from the given link below.**

### _**Let’s Connect**_
<a href="https://linkedin.com/in/mudassiruddin21" target="blank"><img align="center" src="https://cdn.jsdelivr.net/npm/simple-icons@3.0.1/icons/linkedin.svg" alt="mudassiruddin21" height="30" width="40" /></a>

![Connect](https://media1.giphy.com/media/khr2lS27v92PQPD3oa/giphy.gif)

# _**Yes, you now have more knowledge than yesterday, Keep Going.**_
![Happy](https://media2.giphy.com/media/BPJmthQ3YRwD6QqcVD/giphy.gif)
  
