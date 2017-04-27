from __future__ import division, print_function


def nice_hertz_labels(hertz, res=1):
    format = '%0.' + str(res) + 'f'
    out = []
    for freq in hertz:
        if freq >= 1000:
            out.append(format % (freq / 1000) + ' k')
        else:
            out.append('%d' % freq)
    return out
