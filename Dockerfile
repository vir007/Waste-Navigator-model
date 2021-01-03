#builder image
FROM python:3.6-slim-stretch


RUN apt update
RUN apt install -y python3-dev gcc
ADD model.h5 model.h5
#install dependencies
ADD requirements.txt requirements.txt
RUN pip install -r requirements.txt
ADD app.py app.py
#setting up production image

RUN python app.py

EXPOSE 8008

# Start the server
CMD ["python", "app.py", "serve"]
