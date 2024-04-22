from matplotlib import pyplot as plt
from pathlib import Path

PLOT = True

# parse out values based on filter
procedure = "Training"
# procedure = "Testing"
filtered_param = f"{procedure} Network - Time elapsed = "
values = {2**i:0 for i in range(0,7)}

for i, file in enumerate(Path(f'results/main_{procedure.lower()}/').glob('*')):
    print(file, proc:=int(str(file).split('thr')[1].split('.')[0]))
    with open(file, "r") as f: 
        readlines = f.readlines()  
        sampled_lines = [line for line in readlines if filtered_param in line]
        for line in sampled_lines:
            idx = line.find(filtered_param) + len(filtered_param)
            values[proc] = round(float(line[idx:line.find(" ", idx)]),3)
        
print(values)

if PLOT:
    x_values = list(values.keys())
    y_values=list(values.values())    
    plt.plot(x_values, y_values, label=str(file))
    plt.title(f'Strong Scaling {procedure} Time')
    plt.xlabel('Processes')
    plt.ylabel(f'Time (seconds)')
    plt.show()