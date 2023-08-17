import pandas as pd

import campo_continuo as cc

# Read a csv file and return a list of lists using pandas
def read_csv(filename):
    return pd.read_csv(filename, sep=";")

# main
if __name__ == "__main__":
    # Read csv file
    data = read_csv('../../assets/data.csv')

    columnas_continuas_name = ['Previous qualification (grade)', 'Admission grade', 'Unemployment rate', 'Inflation rate', 'GDP']
    columnas_continuas = [cc.CampoContinuo(column_name) for column_name in columnas_continuas_name]
    # Select specific columns and calculate statistics
    column_stats = data[columnas_continuas_name].agg(['mean', 'max', 'min'])

    for column in columnas_continuas:
        column.rangos(column_stats[column.nombre]['max'], column_stats[column.nombre]['mean'], column_stats[column.nombre]['min'])
        #column.print()

    for column in columnas_continuas:
        data[column.nombre] = data[column.nombre].apply(column.calificar)

    print("Saving data to ../../assets/data_preprocessed.csv")
    data.to_csv('../../assets/data_preprocessed.csv', sep=';', index=False)

    print("Separate training and test data 70/30")
    training_data = data.sample(frac=0.7, random_state=0)
    test_data = data.drop(training_data.index)

    print()

    print("Saving training data to ../../assets/training_data.csv")
    training_data.to_csv('../../assets/training_data.csv', sep=';', index=False)

    print("Saving test data to ../../assets/test_data.csv")
    test_data.to_csv('../../assets/test_data.csv', sep=';', index=False)


