# Views

<p align="center">
  <img src="assets/cover.webp" width="700">
</p>

## Overview

**Views** is a Python-based research framework for **macro relative value (RV) analysis**.

The project is designed to:
- Access and analyze market and macro data stored in a **local PostgreSQL (psql) database**
- Run statistical and time-series techniques in Python
- Support systematic and discretionary **macro RV research workflows**

The database is expected to be running locally (e.g. on a Linux tower or workstation), with data ingestion handled separately via dedicated pull scripts.

---

## Key Features

- Local **PostgreSQL-backed data storage**
- Python-first analytics stack (NumPy, Pandas, SciPy, statsmodels)
- Modular utilities for data access, transformations, and analytics
- Designed for macro instruments, spreads, curves, and relative value signals

---

## Project Structure

```text
Views/
├── data/           # local data (ignored by git)
├── data_pull/      # scripts for ingesting data into psql
├── utils/          # reusable analysis utilities
├── main.py         # primary entry point
├── pyproject.toml  # project configuration
└── README.md
