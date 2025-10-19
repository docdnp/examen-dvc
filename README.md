# DVC and Dagshub Exam
In this repository you will find the proposed architecture to implement the exam solution.

```bash       
├── examen_dvc          
│   ├── data       
│   │   ├── processed      
│   │   └── raw       
│   ├── metrics       
│   ├── models      
│   │   ├── data      
│   │   └── models        
│   ├── src       
│   └── README.md.py       
```
Feel free to add folders or files that seem relevant to you.

You must first *Fork* the repo and then clone it to work on it. The deliverable for this exam will be the link to your repository on DagsHub. Make sure to add https://dagshub.com/licence.pedago as a collaborator with read-only rights so it can be graded.

You can download the data through the following link: https://datascientest-mlops.s3.eu-west-1.amazonaws.com/mlops_dvc_fr/raw.csv.


## Run the DVC pipeline

### Running it the very first time and reproduce everything:

```bash
curl https://datascientest-mlops.s3.eu-west-1.amazonaws.com/mlops_dvc_fr/raw.csv -o data/raw_data/raw.csv
uv run dvc repro
```

### Running it with the S3 data from dagshub

Ensure to have logged in with DVC to dagshub's S3:

```bash
uv run dvc remote modify origin --local access_key_id <your-token>
uv run dvc remote modify origin --local secret_access_key <your-token>
```

Then pull latest data and run DVC pipeline (actually nothing should happen):

```bash
uv run dvc pull
uv run dvc repro
```
