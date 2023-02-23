from utils_tesis.signalload import CSV_pandas
import matplotlib.pyplot as plt

signal_name = "I: X0023A-R1A"
# signal_name = "I: X0004A-R2A"
# signal_name = "I: X0071A-R3A"
signals = CSV_pandas()
signals.relay_list()

signal, t, _ = signals.load_data(signal_name)
plt.plot(t, signal)
plt.show()
