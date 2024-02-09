#!/bin/sh

docker build -t eidos-service.di.unito.it/machado/machado-dl4vc-image:latest . -f Dockerfile
docker push eidos-service.di.unito.it/machado/machado-dl4vc-image:latest
docker service rm machado-satdFixed
WANDB_API_KEY=9a53bad34073a4b6bcfa6c2cb67a857e976d86c4  submit --name satdFixed eidos-service.di.unito.it/machado/machado-dl4vc-image:latest  run.sh