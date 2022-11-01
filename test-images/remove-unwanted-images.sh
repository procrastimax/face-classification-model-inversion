#!/bin/bash

for dir in s*
do
    rm -v "$dir"/{2..10}.png
done
