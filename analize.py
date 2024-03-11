from scipy.optimize import curve_fit


class analizer:

    def __init__(self, data, function, initialGuess):
        self.data = data
        self.function = function
        self.initialGuess = initialGuess

        self.time_values = range(len(data))

    def fit(self):
        popt, pcov = curve_fit(self.function, self.time_values, self.data, p0=self.initialGuess)
        print(popt)
        return self.function(self.time_values, *popt), popt
