import numpy as np
import xlwings as xl
import matplotlib.pyplot as plt

from numpy.random import *
from numpy.fft import *

""" Ce script genere des interferogrammes tels qu'obtenus avec un interferometre
de Michelson dans le but d'etudier la transformée de Fourier et de comprendre 
comment la resolution spectrale est déterminée.
"""


def plotCombinedFigures(x, y, w, s, title="", left=400, right=800):
    """"
	On met l'interferogramme et le spectre sur la meme page.
	"""
    fig, (axes, axesFFT) = plt.subplots(2, 1, figsize=(10, 7))
    axes.plot(x, y, '-')
    axes.set_title("Interferogramme")
    axesFFT.plot(w * 1000, abs(s), 'o-')
    axesFFT.set_xlim(left=left, right=right)
    axesFFT.set_xlabel("Longueur d'onde [nm]")
    axesFFT.set_title(title)
    plt.show()


def generateHeNeInterferogram(xMin, xMax, N):
    """ Genere un tableau de N valeurs equidistantes enntre xMin et xMax.
	Ensuite, genere un tableau de N valeurs qui representent un interferogramme
	d'un laser He-Ne a 0.6328 microns. On ajoute du bruit pour rendre le tout
	plus realiste.
	"""
    dx = (xMax - xMin) / N
    x = np.linspace(xMin, xMax, N)
    noise = random(len(x)) * 0.05
    y = 1 + np.cos(2 * np.pi / 0.6328 * x) + noise
    return x, y


def generateWhiteLightInterferogram(xMin, xMax, N):
    """ Genere un tableau de N valeurs equidistantes enntre xMin et xMax.
	Ensuite, genere un tableau de N valeurs qui representent un interferogramme
	d'une source blanche visible. On ajoute du bruit pour rendre le tout
	plus realiste.
	"""
    dx = (xMax - xMin) / N
    x = np.linspace(xMin, xMax, N)
    noise = random(len(x)) * 0.05
    k1 = 1 / 0.4
    k2 = 1 / 0.8
    y = (1 + (np.sin(2 * np.pi * (k1 + k2) * x / 2) / x)) ** 2
    return x, y


def fourierTransformInterferogram(x, y):
    """ À partir du tableau de valeurs Y correspondant a l'abscisse X,
	la transformée de Fourier est calculée et l'axes des fréquences (f en
	µm^-1) et des wavelengths (1/f en microns) est retournée.
	Le spectre est un ensemble de valeurs complexes pour lesquelles l'amplitude
	et la phase sont pertinentes: l'ordre des valeurs commence par la valeur DC (0)
	et monte jusqu'a f_max=1/2/∆x par resolution de ∆f = 1/N/∆x. A partir de la
	(N/2) ieme valeur, la frequence est negative jusqu'a -∆f dans la N-1 case.
	Voir
	https://github.com/dccote/Enseignement/blob/master/HOWTO/HOWTO-Transformes%20de%20Fourier%20discretes.pdf
	"""
    spectrum = fft(y)
    dx = x[1] - x[0]                 # on obtient dx, on suppose equidistant
    N = len(x)                       # on obtient N directement des données
    frequencies = fftfreq(N, dx)     # Cette fonction est fournie par numpy
    wavelengths = 1 / frequencies    # Les fréquences en µm^-1 sont moins utiles que lambda en µm
    return wavelengths, frequencies, spectrum


def readVectorsFromFile(filename):
    print(filename)
    x = np.loadtxt(filename, usecols=0, skiprows=0)
    y = np.loadtxt(filename, usecols=1, skiprows=0)
    return x, y


def readVectorsFromExcelInterferometer(filename,col,rng):
    data = xl.Book(filename).sheets['sheet1']
    x = data.range(f"{col[0]}{rng[0]}:{col[0]}{rng[1]}").value
    y = data.range(f"{col[1]}{rng[0]}:{col[1]}{rng[1]}").value
    x = [float(val.replace(',', '.')) for val in x]
    y = [float(val.replace(',', '.')) for val in y]
    return x, y


def readVectorsFromExcelSensor(filename,col,rng):
    data = xl.Book(filename).sheets['sheet1']
    x = data.range(f"{col[0]}{rng[0]}:{col[0]}{rng[1]}").value
    y = data.range(f"{col[1]}{rng[0]}:{col[1]}{rng[1]}").value
    x = [float(val) for val in x]
    y = [float(val) for val in y]
    return x, y


x_spectrum_hg = np.array(readVectorsFromExcelInterferometer("Hg_spectrum01_mod.xlsx",col=["A","B"],rng=[1,651])[0])  # longueurs d'onde
y_spectrum_hg = np.array(readVectorsFromExcelInterferometer("Hg_spectrum01_mod.xlsx",col=["A","B"],rng=[1,651])[1])  # puissance

plt.plot(x_spectrum_hg, y_spectrum_hg)
plt.show()


""" POWER AND DISPLACEMENT MEASUREMENTS """

x_donnees_hg = np.array(readVectorsFromExcelSensor("Hg_lampe_2.xlsx",col=["D","C"],rng=[19,178])[0])    # temps
y_donnees_hg = np.array(readVectorsFromExcelSensor("Hg_lampe_2.xlsx",col=["D","C"],rng=[19,178])[1])    # tension

# plt.plot(x_donnees_hg, y_donnees_hg)
# plt.show()

data_hg = fourierTransformInterferogram(x_donnees_hg, y_donnees_hg)
wavelengths_hg, frequencies_hg, spectrum_hg = data_hg[0], data_hg[1], data_hg[2]

plt.plot(wavelengths_hg,spectrum_hg)
# plt.show()

# YOOOOOOO

