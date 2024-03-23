import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

from Ploter import ploter
from analize import analizer
from furier import resolveFurier
from main import resolvData, read_csv_file, read_kolumns

# file_paths = [rf"Data\Norm_map_zwobble_{i}_detectorSum_ROI_Cr.csv" for i in range(1, 7, 1)]
file_path_ = r"Data\centers.csv"


# Define the sinusoidal function
def cosuidal_function(t, amplitude, frequency, phase):
    return amplitude * np.cos(2 * np.pi * frequency * t + phase)


def sinusoidal_function(t, amplitude, frequency, phase):
    return amplitude * np.sin(2 * np.pi * frequency * t + phase)


data = np.array(read_kolumns(file_path_), dtype=float)
res = np.zeros_like(data[0])
w = len(data[1:])
print(w)

for d in data[1:2]:
    for i, v in enumerate(d):
        res[i] += v

for d in data[-2:]:
    for i, v in enumerate(d):
        res[-i-1] += v
res /= 2

for file_path, amplitude_values in enumerate((res,)):
    file_path = "Data\\" + str(file_path)+"srednia"
    print(file_path)

    # amplitude_values = resolvData(np.array(read_csv_file(file_path), dtype=float))

    time_values = range(len(amplitude_values))  # Time values

    a, b, c = resolveFurier(amplitude_values).fff()

    fitted_function, popt = analizer(amplitude_values, cosuidal_function, [a, b, c]).fit()

    fitted_amplitude, fitted_frequency, fitted_phase = popt

    fitted_function, popt = analizer(amplitude_values,
                                     lambda x, A, B, C, f=fitted_function: f[x] * (A * x * x + B * x + C),
                                     [0, 0, fitted_amplitude]).fit()

    fitted_function, _ = analizer(amplitude_values,
                                  lambda x, a, b, f=fitted_function: f[x] + a * x + b,
                                  [0, 0]).fit()

    for _ in range(5):
        fft_result = resolveFurier(fitted_function - amplitude_values).fff()

        fitted_function, popt = analizer(amplitude_values,
                                         lambda x, amplitude, frequency, phase, f=fitted_function: f[x]
                                                                                                   + cosuidal_function(
                                             x, amplitude, frequency, phase), [*fft_result]).fit()

    # fitted_function, popt = analizer(amplitude_values,
    #                                 lambda x, A, B, C, f=fitted_function: f[x] * (A * x * x + B * x + C),
    #                                  [0, 0, fitted_amplitude]).fit()

    # Plot original data and fitted function
    plt.figure(figsize=(10, 6))
    plt.plot(time_values, amplitude_values, label='Original Data')
    plt.plot(time_values, fitted_function, label='Fitted Sinusoidal Function', linestyle='--')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.title('Original Data and Fitted Sinusoidal Function')
    plt.legend()
    plt.grid()
    # plt.show()
    plt.savefig(file_path + r".png")
    plt.close()
    plt.cla()
    plt.clf()

    plt.figure(figsize=(10, 6))
    plt.plot(time_values, amplitude_values - fitted_function, label='Original Data')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.title('Residum')
    plt.legend()
    plt.grid()
    # plt.show()
    plt.savefig(file_path + r"residum.png")
    plt.close()
    plt.cla()
    plt.clf()

    print("end data")
