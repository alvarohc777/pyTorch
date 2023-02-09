from utils.signalload import CSV_pandas
import utils.plot_api as plt_api
import matplotlib.pyplot as plt

request_information = {}
request_information["plots"] = {}
signals = CSV_pandas()
signals.relay_list()
request_information["signal_name"] = "I: X0023A-R1A"
request_information["signals"] = signals
request_information["window_length"] = 64
request_information["step"] = 4


# static image
# t, signal, _, _ = plt_api.img_signal(request_information)
# plt.plot(t, signal)
# plt.show()

# static trip
# t, signal, _, _ = plt_api.img_trip(request_information)
# plt.step(t, signal, where="post")
# plt.show()


# static Superimposed
# t, si_signal, _, _ = plt_api.img_si_signal(request_information)
# plt.plot(t, si_signal)
# plt.show()


# Anim signal window
t_windows, signal_windows, max_min, _ = plt_api.anim_signal(request_information)
print(max_min)


# Anim si signal window
# t_windows, si_signal_windows, _, _ = plt_api.anim_si_signal(request_information)
# plt.plot(t_windows[30], si_signal_windows[30])
# plt.show()

# Anim fft
# xf, fft_windows = plt_api.anim_fft(request_information)
# plt.plot(xf, fft_windows[25])
# plt.show()


# Anim si fft
# xf, si_fft_windows = plt_api.anim_si_fft(request_information)
# plt.plot(xf, si_fft_windows[25])
# plt.show()

# Anim trip
# trip_windows = plt_api.anim_trip(request_information)
# t = request_information["t"]
# plt.stem(trip_windows)
# plt.show()

# Anim si trip
# trip_windows = plt_api.anim_si_trip(request_information)
# t = request_information["t"]
# plt.stem(trip_windows)
# plt.show()

print(request_information["plots"].get("img_signal", "not defined"))
print(request_information["plots"].get("img_si_signal", "not defined"))
