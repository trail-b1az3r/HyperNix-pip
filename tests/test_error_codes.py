"""The error code scheme, and the rule that keeps it a scheme.

What these tests are for
------------------------
A code is worth having because you can paste it into a search box and
land somewhere that says what it means. That property holds only while
two things are true: every code is declared in one place, and no two
meanings share a number. Both are easy to lose quietly — the first the
moment somebody writes a code literal at a raise site, the second the
moment two people pick the same round number in the same week.

So most of what follows asserts a refusal, and the registry does its
checking at import rather than when the error is raised, because an
error path is exactly where a latent bug goes unnoticed for a release.
"""
from __future__ import annotations

import pytest

from hypernix.system import errorcatalogue as cat
from hypernix.system.errorcodes import (
    CODES,
    Domain,
    ErrorCode,
    HyperNixError,
    Kind,
    Severity,
    codes_for,
    explain,
    lookup,
    parse,
    raise_for,
    register,
)


class TestTheFormat:
    def test_it_round_trips(self):
        code = ErrorCode(Domain.MODELS, 4, 1015, Kind.RESOURCE,
                         Severity.ERROR, "x")
        assert code.code == "M4-01015.c3"
        again = parse(code.code)
        assert (again.domain, again.tier, again.number, again.kind,
                again.severity) == (Domain.MODELS, 4, 1015, Kind.RESOURCE,
                                    Severity.ERROR)

    def test_the_number_is_five_digits(self):
        """Zero-padded, so codes sort and line up in a log."""
        assert ErrorCode(Domain.SYSTEM, 1, 7, Kind.USAGE,
                         Severity.NOTICE, "x").code == "S1-00007.a1"

    @pytest.mark.parametrize(
        "bad",
        ["", "nope", "M4-1015.c3", "M4-01015.c", "M4-01015.g3",
         "M4-01015.c9", "m4-01015.c3", "M44-01015.c3", "M4_01015.c3"],
    )
    def test_a_malformed_code_is_refused(self, bad):
        with pytest.raises(ValueError):
            parse(bad)

    def test_the_refusal_shows_the_shape(self):
        """Somebody holding a bad code needs the grammar, not "invalid"."""
        with pytest.raises(ValueError) as caught:
            parse("M4-1015")
        assert "L#-NNNNN.kS" in str(caught.value)
        assert "M4-01015.c3" in str(caught.value)

    def test_whitespace_is_forgiven(self):
        """Codes get pasted out of logs with a newline attached."""
        assert parse("  M4-01015.c3\n").number == 1015

    def test_a_tier_outside_one_digit_is_a_bug(self):
        with pytest.raises(ValueError):
            ErrorCode(Domain.MODELS, 42, 1, Kind.USAGE, Severity.ERROR, "x")

    def test_a_number_that_does_not_fit_is_a_bug(self):
        with pytest.raises(ValueError):
            ErrorCode(Domain.MODELS, 1, 100_000, Kind.USAGE, Severity.ERROR, "x")


class TestSeverity:
    def test_it_is_ordered(self):
        """Strings would make every consumer keep the same list in the
        same order. An integer lets a handler say `>= ERROR` and stay
        right for codes added after it was written."""
        assert Severity.NOTICE < Severity.WARNING < Severity.ERROR
        assert Severity.ERROR < Severity.CRITICAL < Severity.FATAL

    def test_a_filter_is_a_comparison(self):
        loud = [c for c in CODES.values() if c.severity >= Severity.CRITICAL]
        assert loud and all(c.severity >= 4 for c in loud)

    def test_it_survives_the_round_trip_as_a_number(self):
        assert parse("S1-00050.c4").severity is Severity.CRITICAL


class TestKind:
    def test_every_kind_letter_is_used_somewhere(self):
        """A kind nothing ever raises is a category that has not earned
        its letter."""
        used = {c.kind for c in CODES.values()}
        assert used == set(Kind)

    def test_kind_says_whose_problem_it_is(self):
        assert cat.RUNNER_NOT_LOADED.kind is Kind.USAGE
        assert cat.RUNTIME_NOT_BUILT.kind is Kind.CONFIG
        assert cat.MODEL_TOO_LARGE.kind is Kind.RESOURCE
        assert cat.RUNNER_DIED.kind is Kind.EXTERNAL
        assert cat.ARCH_MISMATCH.kind is Kind.INTEGRITY
        assert cat.INTERNAL_RUNTIME.kind is Kind.INTERNAL


class TestTheRegistry:
    def test_a_different_code_cannot_take_a_used_number(self):
        with pytest.raises(ValueError):
            register("M", 2, 15, "b", 3, "no model at that path or name")

    def test_two_meanings_cannot_share_a_number(self):
        """The rule that keeps a code searchable. Caught at import, not
        on the error path — an error path is exactly where a latent bug
        sits unnoticed for a release."""
        with pytest.raises(ValueError) as caught:
            register("M", 2, 15, "a", 3, "something else entirely")
        assert "already held by" in str(caught.value)

    def test_the_same_code_twice_is_fine(self):
        """Re-importing a module must not blow up. Only a *different*
        code on a taken number is the collision."""
        first = cat.MODEL_NOT_FOUND
        again = register("M", 2, 15, "a", 3, first.explanation, first.remedy)
        assert again.code == first.code

    def test_the_same_number_in_another_domain_is_fine(self):
        """Numbers are unique per domain, not globally — otherwise every
        domain has to know what every other one has taken."""
        assert cat.RUNNER_NOT_LOADED.number == cat.ELEMENT_LOAD_FAILED.number
        assert cat.RUNNER_NOT_LOADED.code != cat.ELEMENT_LOAD_FAILED.code

    def test_a_code_with_no_explanation_is_refused(self):
        """A code with nothing behind it is a number, and the point of
        the registry is that the number means something everywhere."""
        with pytest.raises(ValueError):
            register("M", 5, 12_345, "a", 3, "   ")

    def test_every_registered_code_has_an_explanation(self):
        bare = [c.code for c in CODES.values() if not c.explanation.strip()]
        assert bare == []

    def test_explanations_are_one_line(self):
        """It goes at the front of a log line. A paragraph there gets
        truncated in exactly the place that mattered."""
        long = [c.code for c in CODES.values() if "\n" in c.explanation]
        assert long == []

    def test_looking_up_an_unregistered_code_names_its_neighbours(self):
        """A mistyped digit is the common way to get here, and a bare
        "unknown code" sends somebody to grep."""
        with pytest.raises(KeyError) as caught:
            lookup("R3-00021.a3")
        assert "R3-" in str(caught.value)

    def test_lookup_rejects_a_malformed_code_before_anything_else(self):
        with pytest.raises(ValueError):
            lookup("definitely not a code")

    def test_codes_for_a_domain_come_back_in_number_order(self):
        numbers = [c.number for c in codes_for(Domain.RUNTIME)]
        assert numbers == sorted(numbers)

    def test_every_domain_letter_is_used(self):
        for domain in Domain:
            assert list(codes_for(domain)), domain


class TestRaising:
    def test_an_unregistered_code_cannot_be_raised(self):
        """Without this the scheme is a convention, and within a release
        you have six spellings of one failure and no way to find out."""
        with pytest.raises(KeyError):
            raise_for("M4-09999.a1")

    def test_the_error_carries_the_code(self):
        with pytest.raises(HyperNixError) as caught:
            raise_for(cat.RUNNER_NOT_LOADED, "nothing in the runner")
        assert caught.value.code is cat.RUNNER_NOT_LOADED
        assert caught.value.severity is Severity.ERROR
        assert caught.value.kind is Kind.USAGE

    def test_the_code_comes_first_in_the_message(self):
        """It is what gets pasted into a search box."""
        with pytest.raises(HyperNixError) as caught:
            raise_for(cat.RUNNER_NOT_LOADED)
        assert str(caught.value).startswith("R3-00020.a3:")

    def test_the_remedy_is_carried_through(self):
        with pytest.raises(HyperNixError) as caught:
            raise_for(cat.RUNTIME_NOT_BUILT)
        assert "hypernix runtime build" in str(caught.value)

    def test_detail_from_the_raise_site_is_kept(self):
        with pytest.raises(HyperNixError) as caught:
            raise_for(cat.MODEL_TOO_LARGE, "needs 14.2 GiB, 8.0 GiB free")
        assert "14.2 GiB" in str(caught.value)

    def test_context_is_structured_not_stringified(self):
        """So a handler can act on it without parsing prose."""
        with pytest.raises(HyperNixError) as caught:
            raise_for(cat.RUNNER_PORT_TAKEN, "8781 is busy", port=8781)
        assert caught.value.to_dict()["context"] == {"port": 8781}

    def test_a_code_string_works_as_well_as_the_object(self):
        with pytest.raises(HyperNixError) as caught:
            raise_for("R3-00020.a3")
        assert caught.value.code is cat.RUNNER_NOT_LOADED

    def test_it_is_catchable_as_one_type(self):
        try:
            raise_for(cat.MODEL_NOT_FOUND)
        except HyperNixError as exc:
            assert exc.code.domain is Domain.MODELS
        else:  # pragma: no cover
            pytest.fail("nothing raised")


class TestTheCatalogue:
    def test_every_name_resolves(self):
        for name in cat.ALL:
            assert cat.by_name(name).code

    def test_an_unknown_name_says_how_many_there_are(self):
        with pytest.raises(KeyError) as caught:
            cat.by_name("no_such_code")
        assert "error code" in str(caught.value)

    def test_explain_works_from_a_bare_string(self):
        """The path for somebody who has a code out of a log and
        nothing else."""
        assert "no model is loaded" in explain("R3-00020.a3")

    def test_explain_does_not_raise_on_an_unknown_code(self):
        """It is a diagnostic. One that throws while you are diagnosing
        is not much use."""
        assert "nothing registers it" in explain("R3-00021.a3")

    def test_deprecated_modules_have_no_codes(self):
        """A code is a promise to keep a meaning stable, and a module on
        its way out is the wrong place to make one."""
        text = (cat.__doc__ or "") + " ".join(
            c.explanation + c.remedy for c in CODES.values()
        )
        assert "pressure_cooker" not in text.replace(
            "``pressure_cooker`` v1–v3 raise what they always raised", ""
        )

    def test_internal_faults_are_severe_and_ours(self):
        for code in (cat.INTERNAL_MODELS, cat.INTERNAL_RUNTIME,
                     cat.INTERNAL_ELEMENTS, cat.INTERNAL_SYSTEM):
            assert code.kind is Kind.INTERNAL
            assert code.severity >= Severity.CRITICAL

    def test_a_notice_is_not_a_failure(self):
        """Severity 1 exists so a module can report something worth
        saying without it reading as an error — `stopped_because` on a
        crawl is the case that prompted it."""
        assert cat.DATA_BUDGET_SPENT.severity is Severity.NOTICE
        assert cat.PROCESS_GONE.severity is Severity.NOTICE


# ---------------------------------------------------------------------------
# The T1 API adopts it
# ---------------------------------------------------------------------------


class TestT1Mapping:
    def test_every_t1_member_has_its_own_code(self):
        """A member added to `T1ErrorCode` without a line in the table
        falls back to INTERNAL_ERROR's code — safe, and wrong. This is
        what makes adding one without the other a red test instead."""
        from hypernix.system.errorcatalogue import T1_CODES
        from hypernix.t1api.errors import T1ErrorCode

        assert {m.name for m in T1ErrorCode} == set(T1_CODES)

    def test_no_two_members_share_a_code(self):
        """1:1, so a search on the new code lands on exactly the meaning
        the old name had."""
        from hypernix.system.errorcatalogue import T1_CODES

        codes = [c.code for c in T1_CODES.values()]
        assert len(codes) == len(set(codes))

    def test_the_error_carries_it(self):
        from hypernix.t1api.errors import T1APIError, T1ErrorCode

        exc = T1APIError(T1ErrorCode.RATE_LIMITED, "slow down")
        assert parse(exc.hx_code).kind is Kind.RESOURCE

    def test_auth_failures_are_the_callers(self):
        from hypernix.system.errorcatalogue import T1_CODES

        for name, code in T1_CODES.items():
            if name.startswith("AUTH_"):
                assert code.kind is Kind.USAGE, name

    def test_the_envelope_keeps_its_old_code_and_adds_the_new_one(self, tmp_path):
        """`code` is a published contract — HyperLink and the waiter TUI
        switch on it. The new code travels *beside* it, never instead."""
        pytest.importorskip("fastapi")
        from fastapi.testclient import TestClient

        from hypernix.security.gatekeeper import Gatekeeper
        from hypernix.security.keymaster import Keymaster
        from hypernix.t1api.app import create_app
        from hypernix.t1api.config import T1APIConfig

        km = Keymaster(store_dir=tmp_path / "km", auto_rotate=False)
        app = create_app(
            config=T1APIConfig(
                token_secret="test-secret-value-that-is-long-enough",
                db_path=str(tmp_path / "t1.sqlite3"),
                module_storage_dir=str(tmp_path / "m"),
                hyperlink_files_dir=str(tmp_path / "f"),
            ),
            keymaster=km,
            gatekeeper=Gatekeeper(keymaster=km, data_dir=tmp_path / "gk",
                                  log_to_file=False),
        )
        got = TestClient(app).get("/usage/current")
        error = got.json()["error"]
        assert error["code"] == "AUTH_MISSING_CREDENTIALS"
        assert error["hx_code"] == "T1-05010.a3"


class TestTheCli:
    def test_explain_a_known_code(self, capsys):
        from hypernix.system.errors_cli import main

        assert main(["explain", "R3-00020.a3"]) == 0
        assert "no model is loaded" in capsys.readouterr().out

    def test_explain_an_unknown_code_exits_nonzero(self, capsys):
        from hypernix.system.errors_cli import main

        assert main(["explain", "R3-00021.a3"]) == 1

    def test_list_filters_by_severity(self, capsys):
        from hypernix.system.errors_cli import main

        main(["list", "--min-severity", "5"])
        out = capsys.readouterr().out
        assert all(".f5" in line or "5  " in line or "fatal" in line
                   for line in out.splitlines()
                   if line and line[0].isalpha() and "-" in line[:4])

    def test_list_one_domain(self, capsys):
        from hypernix.system.errors_cli import main

        main(["list", "R"])
        lines = [ln for ln in capsys.readouterr().out.splitlines() if ln[:1] == "R"]
        assert lines and all(ln.startswith("R") for ln in lines)
