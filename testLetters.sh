#!/usr/bin/env bash
for i in images/letters/*.png; do
echo "$i"
python3 testOCR.py --image "$i"
done