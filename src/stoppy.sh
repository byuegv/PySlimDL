#!/bin/bash
ps aux | grep "python" | awk '{print $2}' | xargs kill -9

rm ./2020*.log
