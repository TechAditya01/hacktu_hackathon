FROM python:3.12-slim

WORKDIR /app

# Copy and install requirements
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy all files into the container
COPY . /home/user/app

# Set environment variables (if needed)
ENV MODEL_NAME=model
ENV MODEL_VERSION=1.0

EXPOSE 8080

# Command to start the Flask app
CMD ["python", "app.py"]
