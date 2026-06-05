import os

ncores = 32

elements_nsi = [1, 2, 3, 4, 5, 6]

print("Process is starting...")

for elem in elements_nsi:

    input = f"sbatch job.sh {ncores} {elem}"
    
    print(f"Launching: {input}")
    
    os.system(input)

print("Did every job!")