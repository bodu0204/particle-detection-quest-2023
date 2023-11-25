FROM jupyter/scipy-notebook

USER root

RUN apt-get update
RUN apt-get install -y python3-dev

RUN pip install --upgrade pip \
 && pip install python-dotenv

COPY ./jupyter_server_config.py ./.jupyter/

COPY ./set_crt.sh ./

RUN bash set_crt.sh

CMD jupyter notebook --allow-root --NotebookApp.token="$NOTEBOOK_PASSWORD" /opt/work