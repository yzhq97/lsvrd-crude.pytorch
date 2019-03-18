#!/usr/bin/env bash

mkdir cache
python scripts/build_dicts.py
python scripts/rel_stat.py
python scripts/generate_balanced_triples.py