FROM ajevnisek/mae:base
RUN pip --no-cache-dir install Cython

WORKDIR /workspace
COPY . /workspace
#COPY requirements.txt /workspace
#
RUN pip --no-cache-dir install -r /workspace/requirements_detr_original.txt
RUN pip install shapely
#COPY . /workspace