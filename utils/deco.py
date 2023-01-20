from decorator import decorator
import time


@decorator
def calc_time(func, *args, **kwargs):
    t0 = time.time()
    func(*args, **kwargs)
    tf = time.time()
    print(f"Total time taken in {func.__name__}: {tf-t0:.5}s")


@decorator
def which_exec(func, *args, **kwargs):
    print(f"Ejecutando {func.__name__}")
    func(*args, **kwargs)
    print(f"Terminó la ejecución de {func.__name__}")


@decorator
def calc_perf(func, *args, **kwargs):
    t0 = time.perf_counter()
    func(*args, **kwargs)
    tf = time.perf_counter()
    print(f"Total time taken in {func.__name__}: {tf-t0:.5}s")


def calculate_time(func):
    def inner1(*args, **kwargs):

        begin = time.time()
        func(*args, **kwargs)
        end = time.time()
        print(f"Total time taken in {func.__name__}: {end-begin}s")

    return inner1


def which_execution(func):
    def inner1(*args, **kwargs):
        print(f"Ejecutando {func.__name__}")
        func(*args, **kwargs)
        print(f"Terminó la ejecución de {func.__name__}")

    return inner1


def time_performance(func):
    def inner1(*args, **kwargs):
        begin = time.perf_counter()
        func(*args, **kwargs)
        end = time.perf_counter()
        print(f"Total time taken in: {func.__name__} => {end-begin}")

    return inner1
