KEY_REF = "K{},{}"
DIODE_REF = "D{},{}"

def get_key_reference(row, col):
    return KEY_REF.format(row, col)

def get_diode_reference(row, col):
    return DIODE_REF.format(row, col)
