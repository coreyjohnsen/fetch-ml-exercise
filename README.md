## Fetch ML Internship Exercise - Predicting 2022 Scan Data

### Summary

This project includes a python script to train a linear regression model based on the data in `data_daily.csv`. This model is then used in the Flask
web app to predict the average number of monthly scans in 2022.

### Training and Model

The training for this project is done using PyTorch. The model is simple linear regression where the input is the date (encoded as days after Jan 1 2021) and the output is the expected number of scans on that date. Linear regression was chosen as the model after graphing the data and observing a strong linear correlation in the data. The model is trained with a learning rate of 0.00002 and SGD as the optimizer, and mean squared error as the loss function. The model is trained with a 70/30 training-test split of the data and is trained over 400,000 epochs.

### Running the Application

#### Running the Application with Docker

This application can be run via a Docker container that can either be built locally or pulled from Docker Hub. After cloning the project and ensuring Docker is installed, the image can be built by running `docker build -t <image_name> .` from the project directory.
Creating the container will do the following:

1. Set the working directory in the image to `/fetch-ml`
2. Install all required packages for the python scripts
3. Copy all files from the project directory to the image
4. Run `python train_linear_model.py`, which will train a linear regression model to fit the data in `data_daily.csv` and save it to `/fetch-ml/models/fetch_2022_linear.pth`
5. Expose port 5000 and define entrypoint

Building the image will take a few minutes due to installing the requirements and training the model. After the image is created, a container can be started with `docker run -p 5000:5000 <image_name>`. This command will start the web server on port 5000 and, after the container is started, the application will be available at [localhost:5000](localhost:5000). To change the desired port for the web server, change `5000:5000` to `<desired_port>:5000`.

To avoid building the image locally, pull the image from the Docker Hub by running `docker pull cjjohnsen/fetch_ml_coreyjohnsen:latest` and then start the container with `docker run -p 5000:5000 cjjohnsen/fetch_ml_coreyjohnsen`. The image download is around 2.5 GB.

#### Running the Application Locally

To run the application locally, first ensure you have python and pip installed, and then run `pip install -r requirements.txt` from the project directory to install necessary dependencies. The model can then be trained and saved with `python train_linear_model.py`. The model should be available in `./models/train_linear_model.py` after this is complete. Finally, the web application can be started by running `python app.py` from the project directory. After this, the application will be available at [localhost:5000](localhost:5000).