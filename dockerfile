# Use the appropriate base image
FROM python:3.10-slim

ENV PYTHONUNBUFFERED True
ENV APP_HOME /app

# Set the working directory
WORKDIR $APP_HOME

# Copy the application code into the container
COPY . ./

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port that the app runs on
EXPOSE 8080

# Run the application using gunicorn
CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 app:app