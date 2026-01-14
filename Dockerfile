FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Set the requirements
COPY requirements.txt .
COPY requirements_dev.txt .

# Install Dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files
COPY . .

COPY .env .

# Expose Port for the app
EXPOSE 5000

# command to run the app
CMD ["bash", "-c", "python src/Pipeline/training_pipeline.py && python app.py"]

