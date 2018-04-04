import os
from pymongo import MongoClient

mongo_url = os.environ.get(key="MONGO_URL", default="mongodb://10.34.163.245:32768")

mongo = MongoClient(mongo_url)
jobs = mongo.quaty.jobs


def update_precision(job_name, precision):
    jobs.update_one({"name": job_name}, {"$set": {"precision": precision}})
