FROM python

ADD requirements.txt /requirements.txt
WORKDIR /
RUN pip install -r requirements.txt
RUN pip install pyfasttext==0.4.4

COPY ./app/. /

CMD ["export", "LC_ALL=en_US.UTF-8"]
CMD ["export", "LANG=en_US.UTF-8"]

EXPOSE 8089

COPY run.sh /run.sh
RUN chmod +x /run.sh
CMD [ "/run.sh" ]
