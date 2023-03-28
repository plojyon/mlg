FROM python:3.8

RUN useradd -ms /bin/bash appuser
USER appuser

WORKDIR /home/appuser/app

COPY src .
COPY requirements.txt requirements.txt

ENV PATH=/home/appuser/.local/bin:$PATH
RUN /usr/local/bin/python3 -m pip install --upgrade pip
RUN pip3 install --user --upgrade setuptools wheel

RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
RUN pip3 install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.0.0+cpu.html
RUN pip3 install -r requirements.txt