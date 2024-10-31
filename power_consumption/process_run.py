import subprocess

for i in range(1, 101):
    subprocess.run(['python', 'main_grasp.py'])
    try:
        with open(f'./data/test_oracle/Shuffle/test_oracle_code.py', "r") as file:
            generated_code = file.read().strip()
    except FileNotFoundError:
        print(f"generated_code not found in iteration {i}.")
        continue
    filename = f'./data/test_oracle/Shuffle/test_oracle_code_process{i}.py'
    with open(filename, 'w') as f:
        f.write(generated_code)
    print(f'Generated {filename}')
print("All processes completed.")
