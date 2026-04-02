
import sys
import os
sys.path.append(os.getcwd())
from skyburst import job_gen
print(dir(job_gen))
if hasattr(job_gen, 'load_processed_jobs'):
    print("Found load_processed_jobs")
else:
    print("Not found")
