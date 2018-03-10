FROM ensemble_nerd_docker
RUN apt-get install -qy python3
RUN apt-get install -qy python3-pip
ADD . /myapp
WORKDIR /myapp
RUN pip3 install -r requirements.txt
RUN pip3 install .
EXPOSE 8000
CMD myapp --port 8000