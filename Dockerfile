# Start from a base image
FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt requirements.txt

# Install the required packages
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Install libgomp1
RUN apt-get update && apt-get install -y libgomp1

# Copy the application code into the container
COPY ./app/app.py /app
# Copy the model object
COPY ./app/flight_delays_lgb_model.pkl /app

# Expose the app port
EXPOSE 8080

# Run command
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]