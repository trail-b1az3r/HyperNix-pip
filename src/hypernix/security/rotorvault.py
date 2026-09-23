"""hypernix.security.rotorvault — the layered cipher behind v2.1 (T2C) keys.

Rotorvault seals bytes under a key in six stages, in this order::

    plaintext
      -> Blowfish (CTR)            a 64-bit Feistel cipher, 1993
      -> Twofish (CTR)             the AES finalist, 1998
      -> rotor stage               Enigma-inspired: stepping rotors and a
                                   plugboard, wired by xoshiro256++
      -> AES-256-GCM               Rijndael, with authentication
      -> inversion                 every bit flipped, byte order reversed
      -> base64url

Every stage has its own key, derived with HKDF-SHA256 from the caller's
key and a fresh random salt, so no two seals share a key even under the
same caller key. :func:`seal_for` puts RSA-OAEP in front, so that anyone
holding a server's *public* key can seal something only that server can
open; :func:`seal_with_password` puts scrypt in front, for a file locked
with a password.

What actually protects the data
-------------------------------
Said plainly, because a cipher with six stages invites the assumption
that it is six times as strong: **the security is AES-256-GCM, keyed by
HKDF.** That stage alone is confidential and authenticated, and every
tampered byte fails there before anything else runs. Blowfish and Twofish
under independent keys add a little defence in depth. The rotor stage is
a keyed byte permutation — historically interesting, cryptographically
weak on its own. The inversion and the base64 are encodings, not
encryption. xoshiro256++ is a fast statistical generator, not a
cryptographic one, so it is used only to wire the rotors from key
material that HKDF already made secret; every random value that must be
unpredictable (salts, keys, secrets) comes from :mod:`secrets`.

The extra stages are here because v2.1 keys were specified with them.
They cost microseconds on a key and milliseconds on a config file, and
none of them can weaken the stage that carries the security.

Blowfish and Twofish are implemented here in pure Python and checked
against their published test vectors (see ``tests/test_rotorvault.py``):
Blowfish has moved to ``cryptography``'s "decrepit" module and Twofish
was never in it. AES-GCM, HKDF and RSA come from ``cryptography``
(``pip install "hypernix[security]"``).
"""
from __future__ import annotations

import base64
import functools
import hashlib
import hmac
import secrets
import struct
from datetime import date

__all__ = [
    "RotorvaultError",
    "Blowfish",
    "Twofish",
    "Xoshiro256PlusPlus",
    "RotorMachine",
    "seal",
    "open_sealed",
    "seal_for",
    "open_sealed_for",
    "seal_with_password",
    "open_with_password",
    "daily_key",
    "generate_rsa_private_key",
    "public_key_pem",
    "public_key_fingerprint",
    "load_private_key",
    "load_public_key",
    "SCHEME",
]

#: The scheme's name, as it appears in key metadata and documentation.
SCHEME = "rotorvault-v1"

_MASK32 = 0xFFFFFFFF
_MASK64 = 0xFFFFFFFFFFFFFFFF


class RotorvaultError(ValueError):
    """A seal that cannot be opened: wrong key, wrong password, or altered."""


# ---------------------------------------------------------------------------
# xoshiro256++ — wires the rotors, never makes secrets
# ---------------------------------------------------------------------------


def _rotl64(x: int, k: int) -> int:
    return ((x << k) | (x >> (64 - k))) & _MASK64


class Xoshiro256PlusPlus:
    """Blackman and Vigna's xoshiro256++ (sometimes written xoroshiro256++).

    Deterministic by design: the same 32-byte seed gives the same stream,
    which is what lets a sealer and an opener build identical rotors.
    That is also why it must never be used for anything secret on its
    own — its output is predictable from its state.
    """

    def __init__(self, seed: bytes | tuple[int, int, int, int]) -> None:
        if isinstance(seed, (bytes, bytearray)):
            if len(seed) != 32:
                raise ValueError("xoshiro256++ takes a 32-byte seed")
            state = list(struct.unpack("<4Q", bytes(seed)))
        else:
            state = [int(v) & _MASK64 for v in seed]
        if len(state) != 4 or not any(state):
            raise ValueError("xoshiro256++ state must be four words, not all zero")
        self._s = state

    def next(self) -> int:
        s = self._s
        result = (_rotl64((s[0] + s[3]) & _MASK64, 23) + s[0]) & _MASK64
        t = (s[1] << 17) & _MASK64
        s[2] ^= s[0]
        s[3] ^= s[1]
        s[1] ^= s[2]
        s[0] ^= s[3]
        s[2] ^= t
        s[3] = _rotl64(s[3], 45)
        return result

    def below(self, bound: int) -> int:
        """Uniform in [0, bound), by rejection rather than a biased modulo."""
        if bound <= 0:
            raise ValueError("bound must be positive")
        limit = _MASK64 + 1 - ((_MASK64 + 1) % bound)
        while True:
            value = self.next()
            if value < limit:
                return value % bound

    def permutation(self, n: int = 256) -> list[int]:
        """A Fisher-Yates shuffle of range(n)."""
        items = list(range(n))
        for i in range(n - 1, 0, -1):
            j = self.below(i + 1)
            items[i], items[j] = items[j], items[i]
        return items


# ---------------------------------------------------------------------------
# Blowfish (Schneier, 1993) — pure Python
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=1)
def _pi_words() -> tuple[int, ...]:
    """The first 1042 32-bit words of pi's fractional part, in hex.

    Blowfish's P-array and S-boxes are exactly these digits. Computed
    with Machin's formula rather than pasted as 8,336 hex digits: a
    table this size is where a transcription error hides, and a formula
    either produces pi or fails the test vectors.
    """
    words = 18 + 4 * 256
    bits = words * 32 + 64
    one = 1 << bits

    def arctan_inverse(x: int) -> int:
        total = term = one // x
        x2 = x * x
        n = 1
        sign = -1
        while term:
            term //= x2
            total += sign * (term // (2 * n + 1))
            sign = -sign
            n += 1
        return total

    pi = 16 * arctan_inverse(5) - 4 * arctan_inverse(239)
    fraction = pi - 3 * one
    digits = fraction >> 64
    return tuple(
        (digits >> (32 * (words - 1 - i))) & _MASK32 for i in range(words)
    )


class Blowfish:
    """Blowfish with a 4- to 56-byte key, one 8-byte block at a time."""

    block_size = 8

    def __init__(self, key: bytes) -> None:
        if not 4 <= len(key) <= 56:
            raise ValueError("a Blowfish key is 4 to 56 bytes")
        words = _pi_words()
        self.p = list(words[:18])
        self.s = [list(words[18 + 256 * i: 18 + 256 * (i + 1)]) for i in range(4)]
        index = 0
        for i in range(18):
            chunk = 0
            for _ in range(4):
                chunk = (chunk << 8) | key[index % len(key)]
                index += 1
            self.p[i] ^= chunk
        left = right = 0
        for i in range(0, 18, 2):
            left, right = self._encrypt_words(left, right)
            self.p[i], self.p[i + 1] = left, right
        for box in self.s:
            for i in range(0, 256, 2):
                left, right = self._encrypt_words(left, right)
                box[i], box[i + 1] = left, right

    def _f(self, x: int) -> int:
        s = self.s
        h = (s[0][x >> 24] + s[1][(x >> 16) & 0xFF]) & _MASK32
        return ((h ^ s[2][(x >> 8) & 0xFF]) + s[3][x & 0xFF]) & _MASK32

    def _encrypt_words(self, left: int, right: int) -> tuple[int, int]:
        p = self.p
        for i in range(16):
            left ^= p[i]
            right ^= self._f(left)
            left, right = right, left
        left, right = right, left
        right ^= p[16]
        left ^= p[17]
        return left, right

    def _decrypt_words(self, left: int, right: int) -> tuple[int, int]:
        p = self.p
        for i in range(17, 1, -1):
            left ^= p[i]
            right ^= self._f(left)
            left, right = right, left
        left, right = right, left
        right ^= p[1]
        left ^= p[0]
        return left, right

    def encrypt_block(self, block: bytes) -> bytes:
        left, right = struct.unpack(">2I", block)
        return struct.pack(">2I", *self._encrypt_words(left, right))

    def decrypt_block(self, block: bytes) -> bytes:
        left, right = struct.unpack(">2I", block)
        return struct.pack(">2I", *self._decrypt_words(left, right))


# ---------------------------------------------------------------------------
# Twofish (Schneier et al., 1998) — pure Python
# ---------------------------------------------------------------------------

_Q_TABLES = (
    (
        (0x8, 0x1, 0x7, 0xD, 0x6, 0xF, 0x3, 0x2, 0x0, 0xB, 0x5, 0x9, 0xE, 0xC, 0xA, 0x4),
        (0xE, 0xC, 0xB, 0x8, 0x1, 0x2, 0x3, 0x5, 0xF, 0x4, 0xA, 0x6, 0x7, 0x0, 0x9, 0xD),
        (0xB, 0xA, 0x5, 0xE, 0x6, 0xD, 0x9, 0x0, 0xC, 0x8, 0xF, 0x3, 0x2, 0x4, 0x7, 0x1),
        (0xD, 0x7, 0xF, 0x4, 0x1, 0x2, 0x6, 0xE, 0x9, 0xB, 0x3, 0x0, 0x8, 0x5, 0xC, 0xA),
    ),
    (
        (0x2, 0x8, 0xB, 0xD, 0xF, 0x7, 0x6, 0xE, 0x3, 0x1, 0x9, 0x4, 0x0, 0xA, 0xC, 0x5),
        (0x1, 0xE, 0x2, 0xB, 0x4, 0xC, 0x3, 0x7, 0x6, 0xD, 0xA, 0x5, 0xF, 0x9, 0x0, 0x8),
        (0x4, 0xC, 0x7, 0x5, 0x1, 0x6, 0x9, 0xA, 0x0, 0xE, 0xD, 0x8, 0x2, 0xB, 0x3, 0xF),
        (0xB, 0x9, 0x5, 0x1, 0xC, 0x3, 0xD, 0xE, 0x6, 0x4, 0x7, 0xF, 0x2, 0x0, 0x8, 0xA),
    ),
)


def _ror4(x: int, n: int) -> int:
    return ((x >> n) | (x << (4 - n))) & 0xF


def _build_q(tables: tuple[tuple[int, ...], ...]) -> tuple[int, ...]:
    t0, t1, t2, t3 = tables
    out = []
    for x in range(256):
        a0, b0 = x >> 4, x & 0xF
        a1 = a0 ^ b0
        b1 = a0 ^ _ror4(b0, 1) ^ ((8 * a0) & 0xF)
        a2, b2 = t0[a1], t1[b1]
        a3 = a2 ^ b2
        b3 = a2 ^ _ror4(b2, 1) ^ ((8 * a2) & 0xF)
        a4, b4 = t2[a3], t3[b3]
        out.append((b4 << 4) | a4)
    return tuple(out)


_Q0 = _build_q(_Q_TABLES[0])
_Q1 = _build_q(_Q_TABLES[1])

_MDS = (
    (0x01, 0xEF, 0x5B, 0x5B),
    (0x5B, 0xEF, 0xEF, 0x01),
    (0xEF, 0x5B, 0x01, 0xEF),
    (0xEF, 0x01, 0xEF, 0x5B),
)
_RS = (
    (0x01, 0xA4, 0x55, 0x87, 0x5A, 0x58, 0xDB, 0x9E),
    (0xA4, 0x56, 0x82, 0xF3, 0x1E, 0xC6, 0x68, 0xE5),
    (0x02, 0xA1, 0xFC, 0xC1, 0x47, 0xAE, 0x3D, 0x19),
    (0xA4, 0x55, 0x87, 0x5A, 0x58, 0xDB, 0x9E, 0x03),
)


def _gf_mult(a: int, b: int, modulus: int) -> int:
    result = 0
    while b:
        if b & 1:
            result ^= a
        a <<= 1
        if a & 0x100:
            a ^= modulus
        b >>= 1
    return result


def _rotl32(x: int, n: int) -> int:
    return ((x << n) | (x >> (32 - n))) & _MASK32


def _rotr32(x: int, n: int) -> int:
    return ((x >> n) | (x << (32 - n))) & _MASK32


def _mds_column(column: int, value: int) -> int:
    word = 0
    for row in range(4):
        word |= _gf_mult(_MDS[row][column], value, 0x169) << (8 * row)
    return word


def _h(x: int, words: list[int]) -> int:
    k = len(words)
    y = [(x >> (8 * i)) & 0xFF for i in range(4)]
    lb = [[(w >> (8 * i)) & 0xFF for i in range(4)] for w in words]
    if k == 4:
        y = [_Q1[y[0]] ^ lb[3][0], _Q0[y[1]] ^ lb[3][1], _Q0[y[2]] ^ lb[3][2], _Q1[y[3]] ^ lb[3][3]]
    if k >= 3:
        y = [_Q1[y[0]] ^ lb[2][0], _Q1[y[1]] ^ lb[2][1], _Q0[y[2]] ^ lb[2][2], _Q0[y[3]] ^ lb[2][3]]
    y = [
        _Q1[_Q0[_Q0[y[0]] ^ lb[1][0]] ^ lb[0][0]],
        _Q0[_Q0[_Q1[y[1]] ^ lb[1][1]] ^ lb[0][1]],
        _Q1[_Q1[_Q0[y[2]] ^ lb[1][2]] ^ lb[0][2]],
        _Q0[_Q1[_Q1[y[3]] ^ lb[1][3]] ^ lb[0][3]],
    ]
    result = 0
    for column in range(4):
        result ^= _mds_column(column, y[column])
    return result


class Twofish:
    """Twofish with a 16-, 24- or 32-byte key, one 16-byte block at a time."""

    block_size = 16

    def __init__(self, key: bytes) -> None:
        if len(key) not in (16, 24, 32):
            raise ValueError("a Twofish key is 16, 24 or 32 bytes")
        k = len(key) // 8
        m = list(struct.unpack(f"<{2 * k}I", key))
        even, odd = m[0::2], m[1::2]
        s_words = []
        for i in range(k):
            chunk = key[8 * i: 8 * i + 8]
            word = 0
            for row in range(4):
                value = 0
                for col in range(8):
                    value ^= _gf_mult(_RS[row][col], chunk[col], 0x14D)
                word |= value << (8 * row)
            s_words.append(word)
        s_words.reverse()
        rho = 0x01010101
        subkeys = []
        for i in range(20):
            a = _h((2 * i * rho) & _MASK32, even)
            b = _rotl32(_h(((2 * i + 1) * rho) & _MASK32, odd), 8)
            subkeys.append((a + b) & _MASK32)
            subkeys.append(_rotl32((a + 2 * b) & _MASK32, 9))
        self.k = subkeys
        # g() depends on the key only through S, so it is tabulated once:
        # four 256-entry tables, one per input byte.
        lb = [[(w >> (8 * i)) & 0xFF for i in range(4)] for w in s_words]
        tables = []
        for position in range(4):
            table = []
            for value in range(256):
                y = value
                if k == 4:
                    y = (_Q1, _Q0, _Q0, _Q1)[position][y] ^ lb[3][position]
                if k >= 3:
                    y = (_Q1, _Q1, _Q0, _Q0)[position][y] ^ lb[2][position]
                first, second, third = (
                    (_Q0, _Q0, _Q1), (_Q1, _Q0, _Q0), (_Q0, _Q1, _Q1), (_Q1, _Q1, _Q0)
                )[position]
                y = third[second[first[y] ^ lb[1][position]] ^ lb[0][position]]
                table.append(_mds_column(position, y))
            tables.append(tuple(table))
        self._g_tables = tuple(tables)

    def _g(self, x: int) -> int:
        t = self._g_tables
        return t[0][x & 0xFF] ^ t[1][(x >> 8) & 0xFF] ^ t[2][(x >> 16) & 0xFF] ^ t[3][x >> 24]

    def encrypt_block(self, block: bytes) -> bytes:
        k = self.k
        r = [w ^ k[i] for i, w in enumerate(struct.unpack("<4I", block))]
        for rnd in range(16):
            t0 = self._g(r[0])
            t1 = self._g(_rotl32(r[1], 8))
            f0 = (t0 + t1 + k[2 * rnd + 8]) & _MASK32
            f1 = (t0 + 2 * t1 + k[2 * rnd + 9]) & _MASK32
            r2 = _rotr32(r[2] ^ f0, 1)
            r3 = _rotl32(r[3], 1) ^ f1
            r = [r2, r3, r[0], r[1]]
        out = [r[2] ^ k[4], r[3] ^ k[5], r[0] ^ k[6], r[1] ^ k[7]]
        return struct.pack("<4I", *out)

    def decrypt_block(self, block: bytes) -> bytes:
        k = self.k
        c = struct.unpack("<4I", block)
        r = [c[2] ^ k[6], c[3] ^ k[7], c[0] ^ k[4], c[1] ^ k[5]]
        for rnd in range(15, -1, -1):
            # Undo the swap, then the round.
            r = [r[2], r[3], r[0], r[1]]
            t0 = self._g(r[0])
            t1 = self._g(_rotl32(r[1], 8))
            f0 = (t0 + t1 + k[2 * rnd + 8]) & _MASK32
            f1 = (t0 + 2 * t1 + k[2 * rnd + 9]) & _MASK32
            r[2] = _rotl32(r[2], 1) ^ f0
            r[3] = _rotr32(r[3] ^ f1, 1)
        return struct.pack("<4I", *(w ^ k[i] for i, w in enumerate(r)))


def _ctr(cipher: Blowfish | Twofish, nonce: bytes, data: bytes) -> bytes:
    """Counter mode: the same call encrypts and decrypts."""
    size = cipher.block_size
    counter_bytes = size - len(nonce)
    out = bytearray()
    for index in range(0, len(data), size):
        counter = (index // size).to_bytes(counter_bytes, "big")
        stream = cipher.encrypt_block(nonce + counter)
        chunk = data[index: index + size]
        out.extend(a ^ b for a, b in zip(chunk, stream, strict=False))
    return bytes(out)


# ---------------------------------------------------------------------------
# The rotor stage — Enigma, loosely
# ---------------------------------------------------------------------------


class RotorMachine:
    """Three stepping 256-position rotors between two passes of a plugboard.

    Like Enigma: each byte passes the plugboard, then each rotor at its
    current offset, then the plugboard again, and the rotors advance
    odometer-fashion after every byte, so the same byte twice in a row
    encrypts differently. Unlike Enigma there is no reflector — the
    reflector is what made Enigma never encrypt a letter to itself, the
    weakness Bletchley exploited — so decryption runs the inverse wiring
    instead of the same path.
    """

    ROTORS = 3

    def __init__(self, seed: bytes) -> None:
        rng = Xoshiro256PlusPlus(seed)
        self.rotors = [rng.permutation() for _ in range(self.ROTORS)]
        self.inverse = []
        for rotor in self.rotors:
            inv = [0] * 256
            for i, v in enumerate(rotor):
                inv[v] = i
            self.inverse.append(inv)
        self.start = [rng.below(256) for _ in range(self.ROTORS)]
        self.notches = [rng.below(256) for _ in range(self.ROTORS)]
        # The plugboard: 64 swapped pairs, an involution.
        order = rng.permutation()
        self.plug = list(range(256))
        for i in range(0, 128, 2):
            a, b = order[i], order[i + 1]
            self.plug[a], self.plug[b] = b, a

    def _step(self, positions: list[int]) -> None:
        for i in range(self.ROTORS):
            positions[i] = (positions[i] + 1) & 0xFF
            if positions[i] != self.notches[i]:
                break

    def encrypt(self, data: bytes) -> bytes:
        positions = list(self.start)
        out = bytearray()
        for byte in data:
            b = self.plug[byte]
            for rotor, pos in zip(self.rotors, positions, strict=True):
                b = (rotor[(b + pos) & 0xFF] - pos) & 0xFF
            out.append(self.plug[b])
            self._step(positions)
        return bytes(out)

    def decrypt(self, data: bytes) -> bytes:
        positions = list(self.start)
        out = bytearray()
        for byte in data:
            b = self.plug[byte]
            for inv, pos in zip(reversed(self.inverse), reversed(positions), strict=True):
                b = (inv[(b + pos) & 0xFF] - pos) & 0xFF
            out.append(self.plug[b])
            self._step(positions)
        return bytes(out)


# ---------------------------------------------------------------------------
# The envelope
# ---------------------------------------------------------------------------

_MAGIC = b"RV1"
_SALT = 16


def _crypto():
    try:
        from cryptography.hazmat.primitives import hashes
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
        from cryptography.hazmat.primitives.kdf.hkdf import HKDF
    except ImportError as exc:  # pragma: no cover - depends on the environment
        from ..errorcatalogue import PACKAGE_MISSING

        raise RotorvaultError(
            f"[{PACKAGE_MISSING}] Rotorvault needs the 'cryptography' package for its "
            'AES-GCM and HKDF stages: pip install "hypernix[security]"'
        ) from exc
    return hashes, AESGCM, HKDF


def _subkeys(key: bytes, salt: bytes, context: bytes) -> dict[str, bytes]:
    hashes, _aes, hkdf_cls = _crypto()
    material = hkdf_cls(
        algorithm=hashes.SHA256(), length=32 * 4 + 4 + 8 + 12,
        salt=salt, info=b"rotorvault/v1|" + context,
    ).derive(key)
    return {
        "blowfish": material[0:32],
        "twofish": material[32:64],
        "rotor": material[64:96],
        "aes": material[96:128],
        "blowfish_nonce": material[128:132],
        "twofish_nonce": material[132:140],
        "aes_nonce": material[140:152],
    }


def _invert(data: bytes) -> bytes:
    return bytes(b ^ 0xFF for b in reversed(data))


def _b64(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _unb64(text: str) -> bytes:
    text = text.strip()
    try:
        return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))
    except (ValueError, TypeError) as exc:
        raise RotorvaultError("not a Rotorvault seal (bad base64)") from exc


def _seal_bytes(key: bytes, plaintext: bytes, context: bytes) -> bytes:
    if len(key) < 16:
        raise ValueError("a Rotorvault key is at least 16 bytes")
    _hashes, aesgcm_cls, _hkdf = _crypto()
    salt = secrets.token_bytes(_SALT)
    k = _subkeys(key, salt, context)
    data = _ctr(Blowfish(k["blowfish"]), k["blowfish_nonce"], plaintext)
    data = _ctr(Twofish(k["twofish"]), k["twofish_nonce"], data)
    data = RotorMachine(k["rotor"]).encrypt(data)
    header = _MAGIC + salt
    data = aesgcm_cls(k["aes"]).encrypt(k["aes_nonce"], data, header + context)
    return header + data


def _open_bytes(key: bytes, blob: bytes, context: bytes) -> bytes:
    _hashes, aesgcm_cls, _hkdf = _crypto()
    if len(blob) < len(_MAGIC) + _SALT + 16 or not blob.startswith(_MAGIC):
        raise RotorvaultError("not a Rotorvault seal")
    header = blob[: len(_MAGIC) + _SALT]
    salt = header[len(_MAGIC):]
    k = _subkeys(key, salt, context)
    try:
        from cryptography.exceptions import InvalidTag
    except ImportError:  # pragma: no cover
        InvalidTag = Exception  # noqa: N806
    try:
        data = aesgcm_cls(k["aes"]).decrypt(k["aes_nonce"], blob[len(header):], header + context)
    except InvalidTag as exc:
        raise RotorvaultError("wrong key, or the seal was altered") from exc
    data = RotorMachine(k["rotor"]).decrypt(data)
    data = _ctr(Twofish(k["twofish"]), k["twofish_nonce"], data)
    return _ctr(Blowfish(k["blowfish"]), k["blowfish_nonce"], data)


def _context(context: str | bytes) -> bytes:
    return context.encode("utf-8") if isinstance(context, str) else bytes(context)


def seal(key: bytes, plaintext: bytes, *, context: str | bytes = b"") -> str:
    """Seal *plaintext* under *key* (at least 16 bytes). Returns text.

    *context* is authenticated but not stored: opening with a different
    context fails, so a seal made for one purpose cannot be replayed as
    another.
    """
    return _b64(_invert(_seal_bytes(key, plaintext, _context(context))))


def open_sealed(key: bytes, token: str, *, context: str | bytes = b"") -> bytes:
    """Open what :func:`seal` made, or raise :class:`RotorvaultError`."""
    return _open_bytes(key, _invert(_unb64(token)), _context(context))


# -- RSA: seal for a public key ---------------------------------------------

_RSA_MAGIC = b"RVR"


def generate_rsa_private_key(bits: int = 3072):
    """A new RSA private key object."""
    _crypto()
    from cryptography.hazmat.primitives.asymmetric import rsa

    return rsa.generate_private_key(public_exponent=65537, key_size=bits)


def public_key_pem(private_or_public) -> str:
    from cryptography.hazmat.primitives import serialization

    public = private_or_public.public_key() if hasattr(private_or_public, "public_key") else private_or_public
    return public.public_bytes(
        serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo
    ).decode("ascii")


def public_key_fingerprint(private_or_public) -> str:
    """SHA-256 of the DER public key, as ``SHA256:<hex, 32 chars>``."""
    from cryptography.hazmat.primitives import serialization

    public = private_or_public.public_key() if hasattr(private_or_public, "public_key") else private_or_public
    der = public.public_bytes(
        serialization.Encoding.DER, serialization.PublicFormat.SubjectPublicKeyInfo
    )
    return "SHA256:" + hashlib.sha256(der).hexdigest()[:32]


def load_private_key(pem: bytes | str):
    _crypto()
    from cryptography.hazmat.primitives import serialization

    data = pem.encode("ascii") if isinstance(pem, str) else pem
    return serialization.load_pem_private_key(data, password=None)


def load_public_key(pem: bytes | str):
    _crypto()
    from cryptography.hazmat.primitives import serialization

    data = pem.encode("ascii") if isinstance(pem, str) else pem
    return serialization.load_pem_public_key(data)


def _oaep():
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.asymmetric import padding

    return padding.OAEP(mgf=padding.MGF1(algorithm=hashes.SHA256()), algorithm=hashes.SHA256(), label=None)


def seal_for(public_key, plaintext: bytes, *, context: str | bytes = b"") -> str:
    """Seal so that only the holder of *public_key*'s private half can open.

    A fresh 32-byte content key is wrapped with RSA-OAEP-SHA256 and the
    plaintext sealed under it.
    """
    _crypto()
    if isinstance(public_key, (str, bytes)):
        public_key = load_public_key(public_key)
    content_key = secrets.token_bytes(32)
    wrapped = public_key.encrypt(content_key, _oaep())
    body = _seal_bytes(content_key, plaintext, _context(context))
    return _b64(_invert(_RSA_MAGIC + struct.pack(">H", len(wrapped)) + wrapped + body))


def open_sealed_for(private_key, token: str, *, context: str | bytes = b"") -> bytes:
    _crypto()
    blob = _invert(_unb64(token))
    if not blob.startswith(_RSA_MAGIC) or len(blob) < len(_RSA_MAGIC) + 2:
        raise RotorvaultError("not a Rotorvault public-key seal")
    (length,) = struct.unpack(">H", blob[3:5])
    wrapped, body = blob[5: 5 + length], blob[5 + length:]
    try:
        content_key = private_key.decrypt(wrapped, _oaep())
    except ValueError as exc:
        raise RotorvaultError("this seal was made for a different key") from exc
    return _open_bytes(content_key, body, _context(context))


# -- scrypt: seal with a password -------------------------------------------

_PW_MAGIC = b"RVP"
_SCRYPT_N, _SCRYPT_R, _SCRYPT_P = 2 ** 15, 8, 1


def _password_key(password: str, salt: bytes, n: int, r: int, p: int) -> bytes:
    return hashlib.scrypt(password.encode("utf-8"), salt=salt, n=n, r=r, p=p,
                          maxmem=128 * 1024 * 1024, dklen=32)


def seal_with_password(password: str, plaintext: bytes, *, context: str | bytes = b"") -> str:
    """Seal under a password, stretched with scrypt (N=2^15, r=8, p=1)."""
    if not password:
        raise ValueError("an empty password protects nothing")
    salt = secrets.token_bytes(16)
    key = _password_key(password, salt, _SCRYPT_N, _SCRYPT_R, _SCRYPT_P)
    params = struct.pack(">BBB", _SCRYPT_N.bit_length() - 1, _SCRYPT_R, _SCRYPT_P)
    return _b64(_invert(_PW_MAGIC + params + salt + _seal_bytes(key, plaintext, _context(context))))


def open_with_password(password: str, token: str, *, context: str | bytes = b"") -> bytes:
    blob = _invert(_unb64(token))
    if not blob.startswith(_PW_MAGIC) or len(blob) < 3 + 3 + 16:
        raise RotorvaultError("not a Rotorvault password seal")
    log_n, r, p = struct.unpack(">BBB", blob[3:6])
    if not (10 <= log_n <= 20 and 1 <= r <= 16 and 1 <= p <= 4):
        raise RotorvaultError("this seal asks for scrypt parameters outside the accepted range")
    salt = blob[6:22]
    key = _password_key(password, salt, 2 ** log_n, r, p)
    try:
        return _open_bytes(key, blob[22:], _context(context))
    except RotorvaultError as exc:
        raise RotorvaultError("wrong password, or the file was altered") from exc


# -- the daily key -----------------------------------------------------------


def daily_key(device_secret: bytes, device_id: str, day: date) -> bytes:
    """The key a device uses on *day* (UTC), derived from its secret.

    HMAC-SHA256 as a PRF over the device and the date: the same secret
    gives a different key every day, a day's key says nothing about the
    next day's, and nobody without the secret can compute either.
    """
    if len(device_secret) < 16:
        raise ValueError("a device secret is at least 16 bytes")
    message = f"rotorvault/day|{device_id}|{day.isoformat()}".encode()
    return hmac.new(device_secret, message, hashlib.sha256).digest()

