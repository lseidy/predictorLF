#!/bin/sh

#docker build -t eidos-service.di.unito.it/machado/machado-dl4vc-image:latest . -f Dockerfile
#docker push eidos-service.di.unito.it/machado/machado-dl4vc-image:latest
#docker service rm machado-satdFixed
#WANDB_API_KEY=9a53bad34073a4b6bcfa6c2cb67a857e976d86c4  submit --name satdFixed16x16_lr-5 eidos-service.di.unito.it/machado/machado-dl4vc-image:latest  run.sh
docker build -t gitlab.di.unito.it:5000/dombrowski/eidos-base-pytorch:1.12.0 . -f Dockerfile
docker push gitlab.di.unito.it:5000/dombrowski/eidos-base-pytorch:1.12.0