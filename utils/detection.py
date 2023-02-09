import numpy as np


def harmonic_distortion(FFT_window, fundamental):

    fault_harmonics = [2, 3, 4]
    caps_harmonics = [5, 6, 7, 8]
    # fundamental =
    HDF = []  # Faults harmonic distortion
    HDC = []  # Caps harmonic distortion
    for harmonic in fault_harmonics:
        component = FFT_window[harmonic] / fundamental
        HDF.append(component)
    for harmonic in caps_harmonics:
        component = FFT_window[harmonic] / fundamental
        HDC.append(component)

    HDF = np.sqrt(np.sum(np.multiply(HDF, HDF)))
    HDC = np.sqrt(np.sum(np.multiply(HDC, HDC)))
    return HDF, HDC


def detector(HDF, HDC, thld=0.1):
    if (HDF > thld) or (HDC > thld):
        if HDF > HDC:
            return 1
        elif HDF < HDC:
            return 2
    return 0


def detection_iter(FFT, fundamental, return_THD=False):
    """
    Iterates through FFT moving windows and returns trip signal

    Parameters
    ----------

    FFT: numpy array
        shape (n, 32)

    fundamnetal: int (optional)
        system's fundamental frequency

    return_THD: boolean (optional)
        wether to return HDF and HDC

    returns
    -------

    TRIP: list
        list of len n

    HDF: list
        list of len n, indices of fault harmonic distortion

    HDC: list
        list of len n, indices of capacitor harmonic distortion
    """

    HDF = []
    HDC = []
    TRIP = []
    print(f"FFT shape: {FFT.shape}")
    # print(f"fundamental: {fundamental.shape}")
    for FFT_window, fundamental in zip(FFT, fundamental):
        hdf, hdc = harmonic_distortion(FFT_window, fundamental)
        TRIP.append(detector(hdf, hdc))

        if return_THD:
            HDF.append(hdf)
            HDC.append(hdc)

    if return_THD:
        return TRIP, HDF, HDC
    return TRIP
