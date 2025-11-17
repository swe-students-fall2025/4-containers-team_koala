FROM python:3.10-slim

# Set working directory inside the container
WORKDIR /app

# Copy only requirements first (for efficient caching)
COPY requirements.txt requirements.txt

# Install dependencies
RUN pip install -r requirements.txt

# Copy the entire web-app directory into the container
COPY . .

# Expose port 5000 
EXPOSE 5000

# Run the Flask app
CMD ["python", "run.py", "database.py"]
