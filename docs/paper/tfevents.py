"""Minimal TensorBoard event reader: no tensorboard/protobuf dependency.
Record: uint64 length, uint32 crc, bytes, uint32 crc. Event proto: 1 wall_time(double) 2 step(int64) 5 summary.
Summary: 1 value(repeated). Value: 1 tag(string) 2 simple_value(float)."""
import struct, glob, os, sys
from collections import defaultdict

def _varint(b, i):
    r = 0; s = 0
    while True:
        c = b[i]; i += 1; r |= (c & 0x7f) << s; s += 7
        if not c & 0x80: return r, i

def _fields(b):
    i = 0; n = len(b)
    while i < n:
        key, i = _varint(b, i); fn, wt = key >> 3, key & 7
        if wt == 0: v, i = _varint(b, i)
        elif wt == 1: v = b[i:i+8]; i += 8
        elif wt == 2: l, i = _varint(b, i); v = b[i:i+l]; i += l
        elif wt == 5: v = b[i:i+4]; i += 4
        else: raise ValueError(wt)
        yield fn, wt, v

def read(path):
    """Return {tag: [(step, value), ...]}"""
    out = defaultdict(list)
    with open(path, 'rb') as f: data = f.read()
    i = 0
    while i + 12 <= len(data):
        (ln,) = struct.unpack('<Q', data[i:i+8]); i += 12
        rec = data[i:i+ln]; i += ln + 4
        step = 0; summary = None
        for fn, wt, v in _fields(rec):
            if fn == 2 and wt == 0: step = v
            elif fn == 5 and wt == 2: summary = v
        if summary is None: continue
        for fn, wt, v in _fields(summary):
            if fn != 1 or wt != 2: continue
            tag = None; val = None
            for fn2, wt2, v2 in _fields(v):
                if fn2 == 1 and wt2 == 2: tag = v2.decode()
                elif fn2 == 2 and wt2 == 5: val = struct.unpack('<f', v2)[0]
            if tag is not None and val is not None: out[tag].append((step, val))
    return out

if __name__ == '__main__':
    for d in sys.argv[1:]:
        for ev in glob.glob(os.path.join(d, 'events.out.tfevents.*')):
            r = read(ev)
            print(d, len(r), 'tags')
            for t in sorted(r): print('  ', t, len(r[t]), 'pts, last step', r[t][-1][0], 'last', round(r[t][-1][1],3))
