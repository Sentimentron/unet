#!/bin/bash

git describe 2> /dev/null || echo 'g'`git rev-parse --short HEAD`
