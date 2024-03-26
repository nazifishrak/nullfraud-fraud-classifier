#!/bin/sh

# Build the Docker image
docker build -t streamlit-app .

# Run the Docker container
docker run -d -p 8501:8501 streamlit-app
