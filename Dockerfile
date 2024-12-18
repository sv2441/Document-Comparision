# Use the official Python image as the base image
FROM python:3.9-slim

# Set environment variables to prevent Python from writing .pyc files and buffering stdout/stderr
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV PORT 8090

# Set the working directory
WORKDIR /app

# Copy the requirements file
COPY requirements.txt /app/

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code to the working directory
COPY . /app/

# Copy the JSON credentials file into the container
COPY kinetic-mile-439413-k0-bc4aef71e45c.json /app/kinetic-mile-439413-k0-bc4aef71e45c.json

# Set environment variables for Google Cloud credentials
ENV GOOGLE_APPLICATION_CREDENTIALS="/app/kinetic-mile-439413-k0-bc4aef71e45c.json"

# Expose the new port
EXPOSE 8090

# Command to run the Streamlit app, binding to the specified port
CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8090", "--server.address=0.0.0.0"]
