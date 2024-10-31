import subprocess

for i in range(1, 101):
    subprocess.run(['python', 'main_grasp.py'])
    # Read the generated_code from the file
    try:
        with open(f'./data/test_oracle/Shuffle/test_oracle_code.py', "r") as file:
            generated_code = file.read().strip()
    except FileNotFoundError:
        print(f"generated_code not found in iteration {i}.")
        continue
    # Construct the filename
    filename = f'./data/test_oracle/Shuffle/test_oracle_code_process{i}.py'
    # Write the generated_code to the new file
    with open(filename, 'w') as f:
        f.write(generated_code)
    print(f'Generated {filename}')
print("All processes completed.")
