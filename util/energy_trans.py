import pandas as pd


def process_and_save_energy_data(input_file, output_file):
    df = pd.read_csv(input_file,encoding='utf-8')
    bins = range(0, 1801, 60)

    df['Minute'] = pd.cut(df['Times'], bins=bins)


    energy_per_minute = df.groupby('Minute')['Total Energy (J)'].sum().reset_index()
    energy_per_minute['Average Energy (W)'] = energy_per_minute['Total Energy (J)'] / 60


    energy_per_minute['Times'] = [int(x.right / 60) for x in energy_per_minute['Minute']]


    energy_per_minute = energy_per_minute[['Times', 'Average Energy (W)']]


    energy_per_minute.to_csv(output_file, index=False)


process_and_save_energy_data('./temp_and_energy_log_VGG16.csv', './Power_VGG16.csv')