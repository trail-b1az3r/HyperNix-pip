"""``--json`` has to mean JSON, on every path that accepts it.

The bug this exists for is quiet and specific. ``--list-tiers`` returned
before it ever looked at ``args.as_json``, so ``steamroller --list-tiers
--json`` printed the human table — a script that asked for machine
output got prose, and found out at ``json.loads``. Both quantiser CLIs
had it, in the same shape, because both grew the listing branch before
they grew the flag.

So the interesting test here is not "does this one command emit JSON".
It is the sweep: for every combination of flags a CLI accepts that
includes ``--json``, the whole of stdout must parse. A flag that is
accepted and ignored is worse than one that is rejected, because the
rejection is visible and the silence is not.
"""
from __future__ import annotations

import contextlib
import io
import json

import pytest

from hypernix.quant import hyprslug_cli, steamroller_cli

#: ``(cli module, argv)`` for every command that takes ``--json`` and
#: needs no model on disk. Anything needing a real GGUF is covered by
#: that tool's own suite; what is being checked here is the flag.
JSON_COMMANDS = [
    pytest.param(steamroller_cli, ["--list-tiers", "--json"], id="steamroller-list-tiers"),
    pytest.param(hyprslug_cli, ["--list-tiers", "--json"], id="hyprslug-list-tiers"),
]


def run(module, argv: list[str]) -> tuple[int, str]:
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = module.main(list(argv))
    return code, out.getvalue()


class TestEveryJsonPathEmitsJson:
    @pytest.mark.parametrize(("module", "argv"), JSON_COMMANDS)
    def test_the_whole_of_stdout_parses(self, module, argv):
        code, text = run(module, argv)
        assert code == 0, text
        assert text.strip(), "--json produced nothing at all"
        # The whole of it, not a line of it: a human table with one JSON
        # line in it would pass a looser check and fail every consumer.
        json.loads(text)

    @pytest.mark.parametrize(("module", "argv"), JSON_COMMANDS)
    def test_it_is_an_object_not_a_bare_scalar(self, module, argv):
        """Room to add a key later without breaking every reader."""
        _code, text = run(module, argv)
        assert isinstance(json.loads(text), dict)

    @pytest.mark.parametrize(("module", "argv"), JSON_COMMANDS)
    def test_the_human_form_is_not_json(self, module, argv):
        """Otherwise the tests above would pass with the flag ignored."""
        without = [flag for flag in argv if flag != "--json"]
        _code, text = run(module, without)
        with pytest.raises(ValueError):
            json.loads(text)


class TestSteamrollerListTiers:
    @pytest.fixture
    def listing(self):
        _code, text = run(steamroller_cli, ["--list-tiers", "--json"])
        return json.loads(text)

    def test_it_carries_the_tiers_the_help_text_names(self, listing):
        from hypernix.quant.steamroller import TIERS

        assert [t["name"] for t in listing["tiers"]] == list(TIERS)

    def test_it_carries_the_sources_the_help_text_names(self, listing):
        from hypernix.quant.steamroller import SOURCE_FORMATS

        assert listing["sources"] == list(SOURCE_FORMATS)

    def test_it_names_the_staging_tier(self, listing):
        """Every descent passes through it, so a caller planning a
        pipeline needs to know which one it is."""
        from hypernix.quant.steamroller import STAGING_TIER

        assert listing["staging_tier"] == STAGING_TIER

    def test_the_extension_tiers_are_marked_as_such(self, listing):
        """The one fact a consumer must not miss: stock llama.cpp
        refuses these files."""
        by_name = {t["name"]: t for t in listing["tiers"]}
        assert by_name["IQ0.5_XXXL"]["upstream"] is False
        assert by_name["Q8_0"]["upstream"] is True
        assert "extension type" in by_name["IQ0.5_XXXL"]["honest_warning"]

    def test_a_tier_that_needs_an_imatrix_says_so(self, listing):
        by_name = {t["name"]: t for t in listing["tiers"]}
        assert by_name["IQ0.5_XXXL"]["needs_imatrix"] is True
        assert by_name["Q8_0"]["needs_imatrix"] is False

    def test_the_human_form_also_names_the_sources(self):
        """It is in the --help epilog and was nowhere in --list-tiers,
        so the two disagreed about what the tool accepts."""
        _code, text = run(steamroller_cli, ["--list-tiers"])
        assert "sources:" in text
        assert "FP16" in text and "Q8_0" in text


class TestHyprslugListTiers:
    @pytest.fixture
    def listing(self):
        _code, text = run(hyprslug_cli, ["--list-tiers", "--json"])
        return json.loads(text)

    def test_both_families_are_present_and_distinguished(self, listing):
        """A caller has to be able to tell which of these produce a file
        any llama.cpp can open."""
        assert all(r["upstream"] for r in listing["recipes"])
        assert not any(t["upstream"] for t in listing["sub_bit_tiers"])

    def test_every_recipe_it_lists_is_one_it_can_write(self, listing):
        from hypernix.quant.hyprslug import resolve_recipe

        for recipe in listing["recipes"]:
            assert resolve_recipe(recipe["name"]) is not None

    def test_a_mix_reports_what_it_widens(self, listing):
        """Q4_K_M is not a block format, and a consumer that treated it
        as one would mis-size every estimate."""
        by_name = {r["name"]: r for r in listing["recipes"]}
        assert by_name["Q4_K_M"]["base"] == "Q4_K"
        assert set(by_name["Q4_K_M"]["overrides"].values()) == {"Q6_K"}
        assert by_name["Q4_K_S"]["overrides"] == {}

    def test_the_sub_bit_tiers_carry_their_ggml_type_ids(self, listing):
        """Deliberately above anything upstream allocates, and the
        number is the reason a stock loader refuses the file by name."""
        ids = {t["name"]: t["ggml_type"] for t in listing["sub_bit_tiers"]}
        assert ids == {
            "IQ0.9_L": 200, "IQ0.75_M": 201, "IQ0.5_XXXL": 202,
            "IQ0.25_UXL": 203, "INT1": 204, "INT4": 205, "FP2": 206,
            "HNX_1375BIT": 207,
        }
        assert all(i >= 200 for i in ids.values())

    def test_the_bit_rates_are_the_real_ones(self, listing):
        from hypernix.quant.lowbit import CODECS
        from hypernix.quant.subbit import PACKINGS

        for tier in listing["sub_bit_tiers"]:
            packing = tier["packing"]
            expected = (
                PACKINGS[packing].bits_per_weight
                if packing in PACKINGS
                else CODECS[packing].bits_per_weight
            )
            assert tier["bits_per_weight"] == pytest.approx(expected)

    def test_each_tier_says_which_family_it_belongs_to(self, listing):
        """There are two now, and they are described by different facts:
        a sign-and-scale packing by how many signs survive, a fixed
        codebook by its levels. A consumer that assumed one shape for
        both crashed on ``KeyError: 'INT4'`` inside this very branch."""
        by_family = {}
        for tier in listing["sub_bit_tiers"]:
            by_family.setdefault(tier["family"], []).append(tier["name"])
        assert set(by_family) == {"sign-and-scale", "fixed-codebook"}
        assert set(by_family["fixed-codebook"]) == {"INT4", "FP2"}
        for tier in listing["sub_bit_tiers"]:
            if tier["family"] == "sign-and-scale":
                assert "signs_kept" in tier and "group" in tier
            else:
                assert "levels" in tier and "code_bits" in tier
