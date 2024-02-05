import numpy as np
import xlwings as xw
import matplotlib.pyplot as plt
import scipy.signal

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
    data = xw.Book(filename).sheets['sheet1']
    x = data.range(f"{col[0]}{rng[0]}:{col[0]}{rng[1]}").value
    y = data.range(f"{col[1]}{rng[0]}:{col[1]}{rng[1]}").value
    z = data.range(f"{col[2]}{rng[0]}:{col[2]}{rng[1]}").value
    x = [float(val) for val in x]
    y = [float(val) for val in y]
    z = [float(val) for val in z]
    return x, y, z


def normalizeVectors(vector):
    return vector - vector[0]


def convertUnits(vector,unit):
    """
    unit is the SI unit used :
    mV to V --> 1e3
    V to mV --> 1e-3
    nm to m --> 1e9
    m to nm --> 1e-9
    """
    return vector/unit


def getMaximumsIndexes(tension_vector):
    return np.array(scipy.signal.argrelextrema(np.array(tension_vector),comparator=np.greater,order=2)).flatten()


def plotGraph(x,y):
    plt.plot(x,y)
    plt.show()


def getWavelength(vector,indexes):
    maxs_mirror_pos_norm = list(vector[indexes])

    pos_norm_diff = []
    for i in range(len(maxs_mirror_pos_norm) - 1):
        pos_norm_diff.append(maxs_mirror_pos_norm[i + 1] - maxs_mirror_pos_norm[i])

    pos_norm_diff = np.array(pos_norm_diff)
    wavelength = convertUnits(np.mean(pos_norm_diff),1e-9)    # wavelength in nm
    print(wavelength)
    return wavelength


def centerVectorAtZero(vector):
    if np.mean(vector) < 0:
        vector = vector - np.mean(vector)
    else:
        vector = vector - np.mean(vector)
    return vector


""" POWER AND DISPLACEMENT MEASUREMENTS FOR HE-NE_2 LASER """

data_he_ne_2 = readVectorsFromExcelSensor("He-Ne_2.xlsx",col=["A","B","C"],rng=[19,520])

time_he_ne = np.array(data_he_ne_2[0])          # temps
mirror_pos_he_ne = np.array(data_he_ne_2[1])    # mirror position
tension_he_ne = np.array(data_he_ne_2[2])       # tension

# normalize vectors
time_norm_he_ne = normalizeVectors(time_he_ne)
mirror_pos_norm_he_ne = normalizeVectors(mirror_pos_he_ne)

# transform vectors to have good units
tension_he_ne = convertUnits(tension_he_ne,1000)                   # mV --> V
time_norm_he_ne = convertUnits(time_norm_he_ne,1000)               # ms --> s
mirror_pos_norm_he_ne = convertUnits(mirror_pos_norm_he_ne,1e6)    # µm --> m

indexes = getMaximumsIndexes(centerVectorAtZero(tension_he_ne))
getWavelength(mirror_pos_norm_he_ne,indexes)

# plotGraph(mirror_pos_norm,tension)


""" POWER AND DISPLACEMENT MEASUREMENTS FOR HE-NE_1 LASER """

data_he_ne_1 = readVectorsFromExcelSensor("He-Ne_1.xlsx",col=["A","B","C"],rng=[19,524])

time = np.array(data_he_ne_1[0])          # time
mirror_pos = np.array(data_he_ne_1[1])    # mirror position
tension = np.array(data_he_ne_1[2])       # tension

# normalize vectors
time_norm = normalizeVectors(time)
mirror_pos_norm = normalizeVectors(mirror_pos)

# transform vectors to have good units
tension = convertUnits(tension,1000)                   # mV --> V
time_norm = convertUnits(time_norm,1000)               # ms --> s
mirror_pos_norm = convertUnits(mirror_pos_norm,1e6)    # µm --> m

indexes = getMaximumsIndexes(centerVectorAtZero(tension))
getWavelength(mirror_pos_norm,indexes)

# plotGraph(mirror_pos_norm,tension)
