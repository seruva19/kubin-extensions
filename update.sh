#!/bin/bash

find . -type f -name '.installed' -exec rm {} +
git pull
