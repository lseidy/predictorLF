#!/bin/sh

docker build -t eidos-service.di.unito.it/machado/machado-dl4vc-image:latest . -f Dockerfile
docker push eidos-service.di.unito.it/machado/machado-dl4vc-image:latest