# Use a base image with Python 3.10
FROM python:3.10-slim

# Set the working directory to /app
WORKDIR /app

# Copy the app directory contents into the container at /app
COPY app/ /app/

# Copy the requirements.txt file from the project directory into the container
COPY requirements.txt /app/

# Install any needed packages specified in requirements.txt
RUN pip install -r requirements.txt

# Expose port 80
EXPOSE 80

# Create the .streamlit directory
RUN mkdir -p ~/.streamlit

# Copy the Streamlit configuration files to the .streamlit directory
COPY app/config.toml ~/.streamlit/config.toml
COPY app/credentials.toml ~/.streamlit/credentials.toml

# Run the application
ENTRYPOINT ["streamlit", "run"]
CMD ["main.py"]
