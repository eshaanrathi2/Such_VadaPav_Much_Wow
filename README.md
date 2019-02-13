## Such_VadaPav_Much_Wow
A CNN classifier with flask web API, to tell if you have a Vada Pav or not. Model trained on a simple CNN and MobileNet having accuracy of 92% and 91% respectively.

### Getting Started :
The project requirements can be installed using
```
pip install -r requirements.txt
```
Also change path in all python files to get the project up and running on your local machine for training and testing.

### Running Tests :
To launch the web API do
```
python app.py
```
Open the web browser at http://localhost:3000/ to upload a sample image and generate predictions. Now, sit back and bask in its glory. :zap:

![Image 1](https://github.com/rohansuresh27/Such_VadaPav_Much_Wow/blob/master/readme/2.png)

To view random model predictions do
```
python predict.py
```
![Image 2](https://github.com/rohansuresh27/Such_VadaPav_Much_Wow/blob/master/readme/7.png)

### Training Model :
Train the model using
```
python train.py
```

### Data Scraping :
Change keyword, path to download images and run using
```
python scraper.py
```









