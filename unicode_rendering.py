import numbers

def superscript(v):
    '''Construct a unicode string containing v as superscript characters.'''
    assert isinstance(v, numbers.Integral)
    chars = []
    if v < 0:
        chars.append(unichr(0x207B))
        v = -v
    for digit in map(int, str(v)):
        if digit == 1:
            chars.append(unichr(0x00B9))
        elif digit == 2:
            chars.append(unichr(0x00B2))
        elif digit == 3:
            chars.append(unichr(0x00B3))
        else:
            chars.append(unichr(0x2070 + digit))
    return ''.join(chars)

def subscript(v):
    '''Construct a unicode string containing v as superscript characters.'''
    assert isinstance(v, numbers.Integral)
    chars = []
    if v < 0:
        chars.append(unichr(0x208B))
        v = -v
    for digit in map(int, str(v)):
        chars.append(unichr(0x2080 + digit))
    return ''.join(chars)


