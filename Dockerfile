# Use an existing docker image as a base
FROM tensorflow/tensorflow:2.11.0-gpu

# Set working directory
WORKDIR /usr/test 

# Copy necessary files into the container
COPY requirements.txt ./

# Install dependencies
RUN apt-get update && \
    # apt-get install -y python3-pip && \
    pip3 install --no-cache-dir -r requirements.txt

# # Expose a port
# EXPOSE 5000

# # Define a command to run the application
# CMD ["python3", "app.py"]
