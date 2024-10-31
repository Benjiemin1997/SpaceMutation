import csv
import pandas as pd

def process_satellite_columns(input_filename, output_filename, columns_to_process):
    with open(input_filename, mode='r', newline='', encoding='utf-8') as infile:
        reader = csv.DictReader(infile)
        with open(output_filename, mode='w', newline='', encoding='utf-8') as outfile:
            fieldnames = reader.fieldnames
            writer = csv.DictWriter(outfile, fieldnames=fieldnames)
            writer.writeheader()
            for row in reader:
                try:

                    for column in columns_to_process:
                        if column in row:

                            satellite_value = eval(row[column])
                            if isinstance(satellite_value, tuple) and len(satellite_value) == 2:
                                extracted_value = satellite_value[1]
                                row[column] = extracted_value
                            else:
                                raise ValueError(f"Invalid tuple format in column {column}")

                    writer.writerow(row)
                except Exception as e:
                    print(f"Error processing row: {row}. Error: {e}")

def add_times_column(input_file, output_file, new_column_name='Times'):
    try:
        df = pd.read_csv(input_file)
    except FileNotFoundError:
        print(f"Error: The file {input_file} was not found.")
        return
    except Exception as e:
        print(f"Error reading the CSV file: {e}")
        return
    num_rows = len(df)

    times_column = pd.Series(range(1, num_rows + 1), name=new_column_name)


    df[new_column_name] = times_column

    try:
        df.to_csv(output_file, index=False)
        print(f"File written successfully to {output_file}")
    except PermissionError:
        print(f"Error: Permission denied when writing to {output_file}. Make sure the file is not open in another program.")
    except FileNotFoundError:
        print(f"Error: Directory for {output_file} does not exist. Please check the path.")
    except Exception as e:
        print(f"An error occurred while writing to the CSV file: {e}")

def sum_satellite_values_by_interval(input_file, output_file, interval=60):
    df = pd.read_csv(input_file)

    if 'Times' not in df.columns:
        raise ValueError("The CSV file must contain a 'Times' column.")
    results = []
    total_seconds = int(df['Times'].max())
    num_intervals = int(total_seconds / interval)
    for i in range(num_intervals):
        start_time = i * interval + 1
        end_time = (i + 1) * interval
        subset = df[(df['Times'] >= start_time) & (df['Times'] < end_time)]
        sum_satellite5 = subset['Satellite5'].sum()
        average_energy = sum_satellite5 / interval
        results.append({
            'Start Time': start_time,
            'End Time': end_time,
            'Average Energy (W)': average_energy
        })
    result_df = pd.DataFrame(results)
    result_df.to_csv(output_file, index=False)
input_file = "./tranpower_Shufflenetv2.csv"
output_file = './tranpower_power_Shufflenetv2.csv'
power_file = './power_trans_stand_Shufflenetv2.csv'
columns_to_process = ['Satellite1', 'Satellite2', 'Satellite3', 'Satellite4', 'Satellite5']
process_satellite_columns(input_file, output_file, columns_to_process)
add_times_column(output_file, output_file)
sum_satellite_values_by_interval(output_file, power_file, interval=60)