#!/bin/bash

git pull
find . -type f -name '.installed' -exec rm {} +
