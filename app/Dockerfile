# Use a base image with Python 3.10
FROM python:3.10-slim

# Set the working directory to /app
WORKDIR /app

# Copy the contents of the app directory to /app
COPY app/ /app/

# Copy the requirements.txt file to /app
COPY requirements.txt /app/

# Install the dependencies
RUN pip install -r requirements.txt

# Expose port 80
EXPOSE 80

# Create the .streamlit directory and copy the configuration files
RUN mkdir -p ~/.streamlit
COPY app/config.toml ~/.streamlit/config.toml
COPY app/credentials.toml ~/.streamlit/credentials.toml

# Run the Streamlit app
ENTRYPOINT ["streamlit", "run"]
CMD ["main.py"]
