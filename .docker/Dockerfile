# Dockerfile for dev container
FROM pytorch/pytorch:2.8.0-cuda12.9-cudnn9-runtime

ARG DEBIAN_FRONTEND=noninteractive

ENV TZ=Europe/Berlin

RUN apt-get update

RUN apt install -y zsh

RUN apt install -y git

WORKDIR "/workspaces/cauvid"