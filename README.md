# ml-pipeline
CI/CD pipeline for ML model for BigData course

## Stack
- Analytics and model training
  - Python 3.x
  - Pandas, NumPy, SkLearn
- Testing
  - unittest + coverage
- Data/Model versioning
  - DVC
- CI/CD
  - Github Actions

## Dataset

Dataset was collected using information about rides from bike sharing system. The goal is to predict number of rents during specific hour based on time, day, season, wheather and etc.

Link: https://www.kaggle.com/competitions/bike-sharing-demand

## Workflow

- Downloaded dataset from Kaggle
- Analyzed and tuned model
- Transformed research notebook into scripts
- Put dataset to S3 using DVC
- Created Dockerfile and docker-compose.yml
- Created piplines usin GitHub Actions
