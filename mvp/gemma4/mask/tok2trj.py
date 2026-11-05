"""tokdocs output (u32 n, then u32 len + i32 tokens) -> TRJ1 with the human id appended as the sacrificial target. Optionally compare to a reference TRJ."""
import sys, struct, numpy as np
src, out, H = sys.argv[1], sys.argv[2], int(sys.argv[3]); ref = sys.argv[4] if len(sys.argv) > 4 else None
b = open(src, 'rb').read(); n = struct.unpack_from('<I', b, 0)[0]; p = 4; docs = []
for _ in range(n):
    (l,) = struct.unpack_from('<I', b, p); p += 4; docs.append(np.frombuffer(b, '<i4', l, p)); p += 4 * l
with open(out, 'wb') as f:
    f.write(b'TRJ1' + struct.pack('<I', n))
    for d in docs:
        seq = np.append(d, H).astype('<i4'); f.write(struct.pack('<II', len(seq), 1) + seq.tobytes() + struct.pack('<I', len(d)))
if ref:
    r = open(ref, 'rb').read(); m = struct.unpack_from('<I', r, 4)[0]; q = 8; same = 0
    for d in docs:
        L, nt = struct.unpack_from('<II', r, q); q += 8; x = np.frombuffer(r, '<i4', L, q); q += 4 * L + 4 * nt
        same += int(len(x) - 1 == len(d) and (x[:-1] == d).all())
    print('identical to reference', same, 'of', n)
print('trj', out, n, 'docs', sum(len(d) + 1 for d in docs), 'tokens')
