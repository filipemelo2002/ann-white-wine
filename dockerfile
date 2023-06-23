# Use the official Python image as the base image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the Python script and the model file to the container
COPY  main.py wine_quality_model.h5 winequality-white.csv /app/

# Install the necessary dependencies
RUN pip install flask flask-cors pandas scikit-learn tensorflow

# Expose the port on which the Flask app will run
EXPOSE 5000

# Set the environment variable for Flask
ENV FLASK_APP=main.py

# Run the Flask app
CMD ["flask", "run", "--host=0.0.0.0"]
