'''Miscellaneous constants for working with raw Beiwe data.

'''


# Mean radius of the Earth (meters)
R_EARTH = 6.371 * (10**6)


# Earth's gravity (meters/second^2)
GEE = 9.80665


# Bytes, see https://physics.nist.gov/cuu/Units/binary.html
BYTES_DEC = {
    'B':  1,
    'KB': 10**3,
    'MB': 10**6,  
    'GB': 10**9,
    'TB': 10**12,
    'PB': 10**15
}
BYTES_BIN = {
    'B':   1,
    'KiB': 2**10,
    'MiB': 2**20,  
    'GiB': 2**30,
    'TiB': 2**40,
    'PiB': 2**50
}
BYTES = {**BYTES_DEC, **BYTES_BIN}