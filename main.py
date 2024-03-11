import numpy as np
import matplotlib.pyplot as plt
import csv
from scipy.optimize import curve_fit
from analize import analizer


# Define the sinusoidal function
def sinusoidal_function(t, amplitude, frequency, phase):
    return amplitude * np.cos(2 * np.pi * frequency * t + phase)


# Replace 'your_file.csv' with the path to your CSV file

def read_csv_file(file_path):
    # Open the CSV file in read mode
    with open(file_path, 'r') as file:
        table = [row for row in csv.reader(file)]

    return table


def gauss(x, sigma, mi):
    return (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mi) / sigma) ** 2)


def resolvData(data):
    return [np.argmax(data[i, :]) for i in range(data.shape[0])]

def resolvDataN(data):
    d = []
    for i in range(data.shape[0]):
        f, rez = analizer(data[i, :], gauss, (1, np.argmax(data[i, :]))).fit()
        d.append(rez[1])
    return d

def plotMaxFromFile(data):
    for i in range(data.shape[0]):
        plt.plot(i, np.argmax(data[i, :]), marker='.', color='b')
    plt.show()

def model_function(x, a, b, c, e):
    return a * np.sin(b * x) + c * np.cos(b * x) + e


def fiting(data):
    # Generate some example data

    x_data = range(data.shape[0])
    y_data = resolvData(data)

    # Fit the model to the data
    initial_guess = [7, 0.25, 7, 30]  # initial guess for the parameters
    params, covariance = curve_fit(model_function, x_data, y_data, p0=initial_guess)

    # Plot the data and the fitted curve
    plt.scatter(x_data, y_data, label='Data')
    plt.plot(x_data, model_function(x_data, *params), color='red', label='Fitted curve')
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Fitting Sin, Cos, and Line to Data')
    plt.show()


def furier(data):
    row = resolvData(data)
    fft_result = np.fft.fft(row)
    # Compute the frequencies
    sampling_frequency = 1  # Sampling frequency
    frequencies = np.fft.fftfreq(len(row), d=1 / sampling_frequency)

    plt.figure()
    plt.plot(frequencies, np.abs(fft_result))  # Plot magnitude of FFT result
    plt.xlabel('Frequency')
    plt.ylabel('Magnitude')
    plt.title('FFT Result')
    plt.grid()
    plt.show()


def plotImshow(data_array):
    # Plot the data as a grayscale image
    plt.imshow(data_array, cmap='gray')

    # Show the plot
    plt.show()


if __name__ == "__main__":
    # Provide the file path of your CSV file
    file_paths = [rf"Data\Norm_map_zwobble_{i}_detectorSum_ROI_Cr.csv" for i in range(1, 7, 1)]
    colors = ["b", "r", "g", "y", "m", "k"]

    plotImshow(np.array(read_csv_file(file_paths[0]), dtype=float))
    plt.show()



