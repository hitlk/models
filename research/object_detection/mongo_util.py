import os
from pymongo import MongoClient

mongo_url = os.environ("MONGO_URL")
if not mongo_url:
    mongo_url = "mongodb://10.34.163.245:32768"

mongo = MongoClient(mongo_url)
jobs = mongo.quaty.jobs


def update_precision(job_name, precision):
    jobs.update({"name": job_name}, {"$set": {"precision": precision}})
