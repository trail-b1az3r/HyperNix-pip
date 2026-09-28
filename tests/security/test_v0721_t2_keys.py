"""T2 keys, SSPKIDs, Host IDs, and T2 to T1 conversion (0.72.1).

The load-bearing property is the round trip: an existing T1 key
presented as a T2 key must convert back to *exactly* the same T1 key, or
it will not authenticate against the store that already holds it. That is
checked over thousands of generated keys rather than one, because the bug
it caught — a special-character alphabet that excluded ``-`` — only fired
on the ~15% of keys that happened to contain one.
"""
from __future__ import annotations

import pytest

from hypernix.security.keymaster import T1KeyGenerator
from hypernix.security.t2keys import (
    ACCESS_LEVELS,
    HOST_ID_LENGTH,
    SSPKID,
    T2S_BODY_LENGTH,
    ServerKeyRegistry,
    SSPKIDCollision,
    T2KeyGenerator,
    T2Type,
    decode_sspkid_index,
    encode_sspkid_index,
    generate_admin_password,
    generate_host_id,
    t2_api_available,
    validate_admin_password,
    validate_host_id,
)


class TestT2Generation:
    def test_a_plain_key_parses_and_carries_its_level(self):
        key = T2KeyGenerator.generate(access_level=5)
        assert key.access_level == 5
        assert key.raw.endswith("-5")
        assert T2KeyGenerator.validate(key.raw)
        assert not key.is_admin

    @pytest.mark.parametrize("level", ACCESS_LEVELS)
    def test_every_access_level_round_trips(self, level):
        key = T2KeyGenerator.generate(access_level=level)
        assert T2KeyGenerator.parse(key.raw).access_level == level

    @pytest.mark.parametrize("level", [0, 10, -1, "x"])
    def test_an_out_of_range_level_is_refused(self, level):
        with pytest.raises(ValueError, match="1-9"):
            T2KeyGenerator.generate(access_level=level)

    def test_admin_keys_carry_a_password_and_say_so(self):
        key = T2KeyGenerator.generate(admin=True)
        assert key.is_admin
        assert 7 <= len(key.password) <= 13
        assert T2KeyGenerator.parse(key.raw).is_admin

    def test_admin_is_the_password_not_the_level(self):
        # A level-9 key without a password is a very privileged *user*
        # key. Conflating the two would let a tier bump grant key
        # management.
        top = T2KeyGenerator.generate(access_level=9)
        assert top.access_level == 9
        assert not top.is_admin

    def test_a_password_without_admin_is_refused(self):
        with pytest.raises(ValueError, match="admin=True"):
            T2KeyGenerator.generate(password="Ab3dEf9")

    def test_a_six_letter_word_is_allowed(self):
        for _ in range(30):
            password = generate_admin_password(include_word=True, length=13)
            ok, reason = validate_admin_password(password)
            assert ok, reason

    def test_t2c_is_not_minted_here(self):
        """A T2C key is sealed, not spelled: this generator points at the
        module and the command that make one (0.72.6)."""
        with pytest.raises(NotImplementedError, match="gkey create -v v2.1"):
            T2KeyGenerator.generate(family=T2Type.T2C)


class TestAdminPasswords:
    def test_length_bounds(self):
        assert not validate_admin_password("Ab3dEf")[0]        # 6
        assert validate_admin_password("Ab3dEf9")[0]           # 7
        assert not validate_admin_password("Ab3dEf9Ab3dEf9")[0]  # 14

    def test_only_the_specified_alphabet(self):
        ok, reason = validate_admin_password("Ab3dEf0")        # 0 is not in 1-9
        assert not ok and "1-9" in reason
        assert not validate_admin_password("Ab3dEf!")[0]

    @pytest.mark.parametrize("bad", ["abc1234", "Zyx9876", "Aaa1bcd"])
    def test_predictable_sequences_are_rejected(self, bad):
        # "securely generated and must not use predictable sequences" is
        # a property of the value; a validator that only checks length
        # lets abc1234 through.
        assert not validate_admin_password(bad)[0]

    def test_single_character_class_is_rejected(self):
        assert not validate_admin_password("QWERTYUIO")[0]

    def test_generated_passwords_always_validate(self):
        for _ in range(300):
            ok, reason = validate_admin_password(generate_admin_password())
            assert ok, reason


class TestT2S:
    def test_body_is_exactly_26(self):
        key = T2KeyGenerator.generate(family=T2Type.T2S, access_level=3)
        assert len(key.body) == T2S_BODY_LENGTH == 26

    def test_a_wrong_length_body_is_refused(self):
        with pytest.raises(ValueError, match="26"):
            T2KeyGenerator.generate(family=T2Type.T2S, body_length=20)

    def test_t2s_can_never_be_admin(self):
        # It is short enough to type, which is precisely why it must not
        # carry administrative authority.
        with pytest.raises(ValueError, match="cannot be an admin"):
            T2KeyGenerator.generate(family=T2Type.T2S, admin=True)

    def test_permissions_outside_hyperlink_are_read_and_write_only(self):
        key = T2KeyGenerator.generate(family=T2Type.T2S, access_level=9)
        assert key.permits("read") and key.permits("write")
        assert not key.permits("admin")

    def test_a_general_t2_key_is_not_restricted_that_way(self):
        key = T2KeyGenerator.generate(admin=True, access_level=9)
        assert key.permits("admin")


class TestConversion:
    def test_t1_to_t2_to_t1_is_exact_over_many_keys(self):
        for _ in range(2000):
            original = T1KeyGenerator.generate()
            wrapped = T2KeyGenerator.from_t1(original, access_level=4)
            assert T2KeyGenerator.to_t1(wrapped) == original

    def test_it_holds_for_keys_containing_a_dash(self):
        # The specific case that broke: '-' is the T2 suffix separator,
        # and excluding it from the special alphabet made these convert
        # to a different key.
        dashed = [
            k for k in (T1KeyGenerator.generate() for _ in range(3000))
            if "-" in T1KeyGenerator.deconstruct(k)["special_chars"]
        ]
        assert dashed, "expected some keys with a dash in their specials"
        for key in dashed:
            assert T2KeyGenerator.to_t1(T2KeyGenerator.from_t1(key, access_level=1)) == key

    def test_a_minted_t2_key_converts_to_a_valid_t1_key(self):
        for _ in range(200):
            key = T2KeyGenerator.generate(admin=True, access_level=7)
            assert T1KeyGenerator.validate(T2KeyGenerator.to_t1(key))

    def test_conversion_is_deterministic(self):
        key = T2KeyGenerator.generate()
        assert T2KeyGenerator.to_t1(key) == T2KeyGenerator.to_t1(key.raw)

    def test_conversion_drops_the_level_and_never_promotes(self):
        key = T2KeyGenerator.generate(admin=True, access_level=9)
        as_t1 = T2KeyGenerator.to_t1(key)
        assert "-9" not in as_t1
        # Coming back the other way never produces an admin key: a
        # format conversion must not be a privilege escalation.
        assert not T2KeyGenerator.from_t1(as_t1, access_level=9).is_admin

    def test_a_t1_key_is_rejected_with_a_pointer(self):
        with pytest.raises(ValueError, match="T1 key"):
            T2KeyGenerator.parse(T1KeyGenerator.generate())

    def test_garbage_is_rejected(self):
        for bad in ("", "T2_", "T2_short-1", "T2_" + "a" * 30, "nonsense"):
            assert not T2KeyGenerator.validate(bad)


class TestSSPKID:
    @pytest.mark.parametrize(
        ("index", "encoded"),
        [(1, "1"), (4, "4"), (5, "!"), (7, "!2"), (10, "?"), (15, "•"),
         (20, "•!"), (25, "*"), (40, "^"), (75, "€"), (100, "$")],
    )
    def test_the_specified_symbols(self, index, encoded):
        assert encode_sspkid_index(index) == encoded
        assert decode_sspkid_index(encoded) == index

    def test_round_trip_and_injectivity_over_a_wide_range(self):
        encoded = [encode_sspkid_index(n) for n in range(1, 20001)]
        assert all(decode_sspkid_index(e) == n for n, e in enumerate(encoded, start=1))
        assert len(set(encoded)) == 20000, "identifier collisions are impossible"

    def test_non_canonical_spellings_are_refused(self):
        # '!!' sums to 10, which is canonically '?'. Accepting both would
        # give one key two identifiers.
        for bad in ("!!", "?!!!!!", "1!", "!5"):
            with pytest.raises(ValueError):
                decode_sspkid_index(bad)

    def test_a_multi_digit_index_says_how_it_is_spelled(self):
        with pytest.raises(ValueError, match="written"):
            decode_sspkid_index("20")

    def test_zero_and_negative_are_refused(self):
        for bad in (0, -1):
            with pytest.raises(ValueError):
                encode_sspkid_index(bad)

    def test_a_bare_server_id_is_not_an_sspkid(self):
        assert not SSPKID.is_sspkid("00042-C1")
        assert SSPKID.is_sspkid("00042-C1#3")
        with pytest.raises(ValueError, match="V1 Server ID"):
            SSPKID.parse("00042-C1")

    def test_a_malformed_server_id_is_refused(self):
        with pytest.raises(ValueError, match="V1 Server ID"):
            SSPKID(server_id="not-a-server", index=1)


class TestServerKeyRegistry:
    def test_many_keys_on_one_v1_server(self):
        registry = ServerKeyRegistry(store_dir=None)
        for i in range(5):
            registry.allocate(f"key{i}", "00042-C1")
        assert len(registry.keys_on("00042-C1")) == 5
        assert {str(registry.sspkid_for(f"key{i}")) for i in range(5)} == {
            "00042-C1#1", "00042-C1#2", "00042-C1#3", "00042-C1#4", "00042-C1#!"
        }

    def test_one_key_per_sspkid(self):
        registry = ServerKeyRegistry(store_dir=None)
        registry.assign("keyA", SSPKID("00042-C1", 2))
        with pytest.raises(SSPKIDCollision, match="already assigned"):
            registry.assign("keyB", SSPKID("00042-C1", 2))

    def test_reassigning_the_same_key_is_not_a_collision(self):
        registry = ServerKeyRegistry(store_dir=None)
        registry.assign("keyA", SSPKID("00042-C1", 2))
        registry.assign("keyA", SSPKID("00042-C1", 2))
        assert registry.resolve("00042-C1#2") == "keyA"

    def test_rehoming_releases_the_old_identifier(self):
        registry = ServerKeyRegistry(store_dir=None)
        registry.assign("keyA", SSPKID("00042-C1", 2))
        registry.assign("keyA", SSPKID("00042-C1", 3))
        assert registry.resolve("00042-C1#2") is None
        assert registry.resolve("00042-C1#3") == "keyA"

    def test_allocation_reuses_the_lowest_free_index(self):
        registry = ServerKeyRegistry(store_dir=None)
        for i in range(3):
            registry.allocate(f"key{i}", "00042-C1")
        registry.release("key1")
        assert str(registry.allocate("newkey", "00042-C1")) == "00042-C1#2"

    def test_resolution_finds_the_right_key(self):
        registry = ServerKeyRegistry(store_dir=None)
        registry.allocate("first", "00001-A1")
        registry.allocate("second", "00001-A1")
        assert registry.resolve("00001-A1#2") == "second"
        assert registry.resolve("00001-A1#9") is None


class TestHostID:
    def test_generated_host_ids_validate(self):
        for _ in range(200):
            host_id = generate_host_id()
            assert len(host_id) == HOST_ID_LENGTH == 54
            assert validate_host_id(host_id)

    def test_a_host_id_is_not_a_server_id_or_an_sspkid(self):
        # Distinguishable by construction, which is what lets waiter -F
        # decide what it was given rather than guess.
        host_id = generate_host_id()
        assert "#" not in host_id
        assert not SSPKID.is_sspkid(host_id)
        assert not validate_host_id("00042-C1")
        assert not validate_host_id("00042-C1#3")

    def test_wrong_length_is_refused(self):
        assert not validate_host_id("a" * 53 + "!" + "x")
        assert not validate_host_id("a" * 52 + "!")


class TestReleaseGating:
    def test_the_t2_api_is_not_released_in_the_0_x_line(self):
        assert not t2_api_available("0.72.1")
        assert not t2_api_available("0.99.9")

    def test_it_arrives_in_1_x(self):
        assert t2_api_available("1.0.0")
        assert t2_api_available("1.2.0")
