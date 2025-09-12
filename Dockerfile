# windows:
# docker build -t ds_beneficiaries:latest . && docker run --rm -it -v "%CD%":/usr/local/wwf_es_beneficiaries ds_beneficiaries:latest
# linux/mac:
# docker build -t ds_beneficiaries:latest . && docker run --rm -it -v `pwd`:/usr/local/wwf_es_beneficiaries ds_beneficiaries:latest
FROM python:3.11-slim

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

ARG WORKDIR=/usr/local/wwf_es_beneficiaries
ENV WORKDIR=${WORKDIR}

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    ca-certificates \
    gdal-bin \
    libgdal-dev \
    proj-bin \
    libproj-dev \
    libgeos-dev \
    libsqlite3-dev \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --upgrade pip setuptools wheel
WORKDIR ${WORKDIR}

COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

ARG CACHEBUST=1
#ARG ECOSHARD_COMMIT=f9ccf00
RUN git clone https://github.com/springinnovate/ecoshard.git /usr/local/ecoshard && \
    cd /usr/local/ecoshard && \
    #git checkout ${ECOSHARD_COMMIT} && \
    pip install . && \
    git log -1 --format='%h on %ci' > /usr/local/ecoshard.gitversion


COPY ./shortest_distances /usr/local/shortest_distances
RUN pip install /usr/local/shortest_distances

RUN useradd -ms /bin/bash user && chown -R user:user ${WORKDIR} /usr/local/ecoshard /usr/local/shortest_distances
USER user

CMD ["/bin/bash"]
