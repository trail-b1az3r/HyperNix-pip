"""Rotorvault — the layered cipher behind v2.1 (T2C) keys and ``waiter serv -e``.

Blowfish and Twofish are implemented in pure Python, so the first thing
checked is that they are the ciphers they claim to be: the published
test vectors. A home-made cipher that round-trips is not evidence of
anything — encrypt and decrypt can be wrong in matching ways.

Then the envelope: what it seals it can open, what it did not seal (or
sealed for another purpose, key or password) it refuses, and one flipped
byte anywhere is caught.
"""
from __future__ import annotations

import datetime
import os

import pytest

from hypernix.security import rotorvault as rv

# ---------------------------------------------------------------------------
# The primitives are the real ones
# ---------------------------------------------------------------------------


class TestBlowfishVectors:
    """Eric Young's Blowfish vectors (the ones Schneier's page links)."""

    @pytest.mark.parametrize("key,plain,cipher", [
        ("0000000000000000", "0000000000000000", "4ef997456198dd78"),
        ("ffffffffffffffff", "ffffffffffffffff", "51866fd5b85ecb8a"),
        ("3000000000000000", "1000000000000001", "7d856f9a613063f2"),
        ("0123456789abcdef", "1111111111111111", "61f9c3802281b096"),
        ("fedcba9876543210", "0123456789abcdef", "0aceab0fc6a0a28d"),
    ])
    def test_vector(self, key, plain, cipher):
        bf = rv.Blowfish(bytes.fromhex(key))
        assert bf.encrypt_block(bytes.fromhex(plain)).hex() == cipher
        assert bf.decrypt_block(bytes.fromhex(cipher)).hex() == plain

    def test_pi_is_pi(self):
        """The P-array starts with pi's first hex digits after the point."""
        assert rv._pi_words()[:4] == (0x243F6A88, 0x85A308D3, 0x13198A2E, 0x03707344)

    def test_key_length_is_checked(self):
        with pytest.raises(ValueError):
            rv.Blowfish(b"abc")


class TestTwofishVectors:
    """The zero-key vectors from the Twofish paper's ecb_tbl.txt."""

    @pytest.mark.parametrize("size,cipher", [
        (16, "9f589f5cf6122c32b6bfec2f2ae8c35a"),
        (24, "efa71f788965bd4453f860178fc19101"),
        (32, "57ff739d4dc92c1bd7fc01700cc8216f"),
    ])
    def test_zero_key(self, size, cipher):
        tf = rv.Twofish(bytes(size))
        assert tf.encrypt_block(bytes(16)).hex() == cipher
        assert tf.decrypt_block(bytes.fromhex(cipher)) == bytes(16)

    def test_the_second_vector_chains(self):
        """ecb_tbl.txt I=2: the key is 0, the plaintext is I=1's output."""
        tf = rv.Twofish(bytes(16))
        assert tf.encrypt_block(bytes.fromhex("9f589f5cf6122c32b6bfec2f2ae8c35a")).hex() == \
            "d491db16e7b1c39e86cb086b789f5419"

    def test_decrypt_inverts_encrypt_under_a_random_key(self):
        tf = rv.Twofish(os.urandom(32))
        block = os.urandom(16)
        assert tf.decrypt_block(tf.encrypt_block(block)) == block


class TestXoshiro:
    def test_the_reference_first_outputs(self):
        """State {1, 2, 3, 4}: rotl(1 + 4, 23) + 1 = 41943041."""
        rng = rv.Xoshiro256PlusPlus((1, 2, 3, 4))
        assert [rng.next() for _ in range(2)] == [41943041, 58720359]

    def test_same_seed_same_stream(self):
        seed = os.urandom(32)
        a, b = rv.Xoshiro256PlusPlus(seed), rv.Xoshiro256PlusPlus(seed)
        assert [a.next() for _ in range(5)] == [b.next() for _ in range(5)]

    def test_all_zero_state_is_refused(self):
        with pytest.raises(ValueError):
            rv.Xoshiro256PlusPlus(bytes(32))

    def test_permutation_is_a_permutation(self):
        assert sorted(rv.Xoshiro256PlusPlus(os.urandom(32)).permutation()) == list(range(256))


class TestRotorMachine:
    def test_round_trip(self):
        machine = rv.RotorMachine(os.urandom(32))
        data = os.urandom(3000)
        assert machine.decrypt(machine.encrypt(data)) == data

    def test_repeated_bytes_do_not_repeat(self):
        """The rotors step: the property that made Enigma more than a
        substitution cipher."""
        out = rv.RotorMachine(os.urandom(32)).encrypt(b"A" * 64)
        assert len(set(out)) > 16


# ---------------------------------------------------------------------------
# The envelope
# ---------------------------------------------------------------------------

def _has_cryptography() -> bool:
    try:
        import cryptography.hazmat.primitives.ciphers.aead  # noqa: F401
    except ImportError:
        return False
    return True


#: The envelope's AES-GCM, HKDF and RSA stages come from `cryptography`;
#: the ciphers above do not, and are checked either way.
needs_crypto = pytest.mark.skipif(not _has_cryptography(), reason='needs "hypernix[security]"')


@needs_crypto
class TestSeal:
    def test_round_trip(self):
        key = os.urandom(32)
        for data in (b"", b"x", os.urandom(1000)):
            assert rv.open_sealed(key, rv.seal(key, data)) == data

    def test_two_seals_of_the_same_bytes_differ(self):
        key = os.urandom(32)
        assert rv.seal(key, b"same") != rv.seal(key, b"same")

    def test_the_wrong_key_is_refused(self):
        token = rv.seal(os.urandom(32), b"secret")
        with pytest.raises(rv.RotorvaultError):
            rv.open_sealed(os.urandom(32), token)

    def test_the_context_is_bound(self):
        key = os.urandom(32)
        token = rv.seal(key, b"secret", context="config")
        with pytest.raises(rv.RotorvaultError):
            rv.open_sealed(key, token, context="key")

    def test_every_flipped_byte_is_caught(self):
        """AES-GCM is the stage that authenticates; nothing it covers can
        change unnoticed, and the header is authenticated as data."""
        key = os.urandom(32)
        blob = rv._invert(rv._unb64(rv.seal(key, b"secret payload")))
        for index in range(len(blob)):
            damaged = bytearray(blob)
            damaged[index] ^= 0x01
            with pytest.raises(rv.RotorvaultError):
                rv._open_bytes(key, bytes(damaged), b"")

    def test_the_output_is_the_inverted_envelope(self):
        """Inversion and base64 are the last two stages, in that order."""
        key = os.urandom(32)
        raw = rv._invert(rv._unb64(rv.seal(key, b"x")))
        assert raw.startswith(b"RV1")

    def test_a_short_key_is_refused(self):
        with pytest.raises(ValueError):
            rv.seal(b"short", b"x")

    def test_garbage_is_refused_not_crashed_on(self):
        for junk in ("", "!!!", rv._b64(b"RV1"), "A" * 10):
            with pytest.raises(rv.RotorvaultError):
                rv.open_sealed(os.urandom(32), junk)


@needs_crypto
class TestPublicKeySeal:
    @pytest.fixture(scope="class")
    def keypair(self):
        return rv.generate_rsa_private_key(2048)

    def test_only_the_private_half_opens(self, keypair):
        token = rv.seal_for(keypair.public_key(), b"T2_key")
        assert rv.open_sealed_for(keypair, token) == b"T2_key"
        with pytest.raises(rv.RotorvaultError):
            rv.open_sealed_for(rv.generate_rsa_private_key(2048), token)

    def test_pem_round_trip(self, keypair):
        pem = rv.public_key_pem(keypair)
        token = rv.seal_for(pem, b"x")
        assert rv.open_sealed_for(keypair, token) == b"x"
        assert rv.public_key_fingerprint(rv.load_public_key(pem)) == rv.public_key_fingerprint(keypair)


@needs_crypto
class TestPasswordSeal:
    def test_round_trip_and_wrong_password(self):
        token = rv.seal_with_password("correct horse", b"config")
        assert rv.open_with_password("correct horse", token) == b"config"
        with pytest.raises(rv.RotorvaultError, match="wrong password"):
            rv.open_with_password("battery staple", token)

    def test_an_empty_password_is_refused(self):
        with pytest.raises(ValueError):
            rv.seal_with_password("", b"x")


@needs_crypto
class TestDailyKey:
    def test_it_changes_each_day_and_per_device(self):
        secret = os.urandom(32)
        day = datetime.date(2026, 9, 23)
        today = rv.daily_key(secret, "dev", day)
        assert today == rv.daily_key(secret, "dev", day)
        assert today != rv.daily_key(secret, "dev", day + datetime.timedelta(days=1))
        assert today != rv.daily_key(secret, "other", day)
        assert today != rv.daily_key(os.urandom(32), "dev", day)
