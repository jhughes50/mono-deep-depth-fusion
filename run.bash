#!/bin/bash

docker run --rm -it --gpus '"device=0"' \
    --name mono-deep-depth-fusion \
    --network=host \
    -p 8116:8125 \
    -e "TERM=xterm-256color" \
    -e DISPLAY=$DISPLAY \
    -e QT_X11_NO_MITSHM=1 \
    -e XAUTHORITY=$XAUTH \
    -v "./loaders:/home/`whoami`/loaders" \
    -v "./utils:/home/`whoami`/utils" \
    -v "./models:/home/`whoami`/models" \
    mono-dense-depth-fusion:dev \
    bash
