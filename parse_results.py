# Read from a directory of results files recursively and count how many files contains "success" in the file name
import os

dirname = "results_cube/oppo_side/first_20"
success_count = 0
file_count = 0
for root, dirs, files in os.walk(dirname):
    for file in files:
        if "success" in file:
            success_count += 1
    
        if "success" in file or "failure" in file:
            file_count += 1

print(f"Found {success_count} success files out of {file_count} total files")