#python image
FROM python:3.11-slim

#Install espeak-ng and other system dependencies
RUN apt-get update && \
    apt-get install -y espeak-ng && \
    rm -rf /var/lib/apt/lists/*

#working directory of the container
WORKDIR /app

#copy requirements
COPY requirements.txt .

#install requirements
RUN pip install --no-cache-dir -r requirements.txt

#copy distribution folder to container
COPY dist /app/dist

#install core from container
RUN pip install /app/dist/core-0.0.0-py3-none-any.whl

#copy core files to container
COPY core /app/core

#expose directory to python
ENV PYTHONPATH="/app"

#ports, can change, use if necessary
EXPOSE 5000  

# Set runtime env variables (optional)
ENV PYTHONUNBUFFERED=1

#application gateway
CMD ["python", "core/app.py"]
