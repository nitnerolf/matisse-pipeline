from pathlib import Path

import pytest
from typer.testing import CliRunner

from matisse.cli.main import app

runner = CliRunner()


@pytest.mark.parametrize("flag", ["--macports-probe", "--no-macports-probe"])
def test_doctor_command_runs(flag):
    """Ensure 'matisse doctor' command runs without crashing."""
    result = runner.invoke(
        app,
        ["doctor", flag],
        catch_exceptions=False,
    )
    # Exit code can be 0 (success) or 2 (esorex not found/configured)
    # This test just ensures the command runs
    assert result.exit_code in (0, 2), f"Unexpected exit code: {result.exit_code}"


def test_doctor_command_with_verbose():
    """Ensure 'matisse doctor' with verbose flag runs."""
    result = runner.invoke(
        app,
        ["doctor", "--verbose", "--no-macports-probe"],
        catch_exceptions=False,
    )
    assert result.exit_code in (0, 2), f"Unexpected exit code: {result.exit_code}"


def test_doctor_command_with_no_require_any():
    """Ensure 'matisse doctor' with --no-require-any flag runs."""
    result = runner.invoke(
        app,
        ["doctor", "--no-require-any", "--no-macports-probe"],
        catch_exceptions=False,
    )
    assert result.exit_code in (0, 2), f"Unexpected exit code: {result.exit_code}"


def test_doctor_command_esorex_not_found(monkeypatch):
    """Ensure 'matisse doctor' handles missing esorex gracefully."""
    from matisse.cli import doctor as doctor_module

    # Mock shutil.which to return None (esorex not found)
    monkeypatch.setattr(doctor_module.shutil, "which", lambda cmd: None)

    result = runner.invoke(
        app,
        ["doctor"],
        catch_exceptions=False,
    )
    # Should exit with code 2 (fatal error)
    assert result.exit_code == 2, f"Unexpected exit code: {result.exit_code}"
    # Should mention esorex not found
    assert "esorex" in result.output.lower()


def test_doctor_reports_calibrator_database_status(monkeypatch):
    """Ensure 'matisse doctor' reports calibrator database status."""
    from matisse.cli import doctor as doctor_module

    monkeypatch.setattr(
        doctor_module,
        "database_status",
        lambda: {
            "vBoekelDatabase.fits": "cached",
            "calib_spec_db_v10.fits": "cached",
            "calib_spec_db_v10_supplement.fits": "missing",
        },
    )
    monkeypatch.setattr(
        doctor_module,
        "check_esorex_in_path",
        lambda: doctor_module.CheckResult("esorex", True, "found at /mock/esorex"),
    )
    monkeypatch.setattr(
        doctor_module,
        "check_matisse_recipes",
        lambda recipe_dir, require_any: (
            doctor_module.CheckResult("MATISSE recipes", True, "1 found."),
            ["mat_raw_estimates"],
        ),
    )

    result = runner.invoke(
        app,
        ["doctor", "--no-macports-probe"],
        catch_exceptions=False,
    )

    assert result.exit_code == 0
    assert "calibrator databases" in result.output
    assert "missing 1/3" in result.output


# ---------- doctor unit tests (fully mocked, no esorex needed) ----------


class TestGetEnvRecipeDirs:
    """Unit tests for _get_env_recipe_dirs."""

    def test_empty_env(self, monkeypatch):
        from matisse.cli import doctor as doctor_module

        monkeypatch.delenv("ESOREX_PLUGIN_DIR", raising=False)
        assert doctor_module._get_env_recipe_dirs() == []

    def test_single_dir(self, monkeypatch):
        from matisse.cli import doctor as doctor_module

        monkeypatch.setenv("ESOREX_PLUGIN_DIR", "/opt/local/lib/esopipes-plugins")
        result = doctor_module._get_env_recipe_dirs()
        assert len(result) == 1
        assert result[0] == Path("/opt/local/lib/esopipes-plugins")

    def test_multiple_dirs(self, monkeypatch):
        from matisse.cli import doctor as doctor_module

        monkeypatch.setenv("ESOREX_PLUGIN_DIR", "/path/a:/path/b:/path/c")
        result = doctor_module._get_env_recipe_dirs()
        assert len(result) == 3
        assert result[1] == Path("/path/b")

    def test_whitespace_only_env(self, monkeypatch):
        from matisse.cli import doctor as doctor_module

        monkeypatch.setenv("ESOREX_PLUGIN_DIR", "   ")
        assert doctor_module._get_env_recipe_dirs() == []


class TestCandidateRecipeDirs:
    """Unit tests for _candidate_recipe_dirs."""

    def test_deduplicates(self, monkeypatch):
        from matisse.cli import doctor as doctor_module

        monkeypatch.delenv("ESOREX_PLUGIN_DIR", raising=False)
        dup = Path("/opt/local/lib/esopipes-plugins")
        result = doctor_module._candidate_recipe_dirs([dup])
        paths_str = [str(p) for p in result]
        assert paths_str.count(str(dup)) == 1

    def test_env_dirs_come_first(self, monkeypatch):
        from matisse.cli import doctor as doctor_module

        monkeypatch.setenv("ESOREX_PLUGIN_DIR", "/my/custom/dir")
        result = doctor_module._candidate_recipe_dirs([])
        assert result[0] == Path("/my/custom/dir")

    def test_extra_candidates_appended(self, monkeypatch):
        from matisse.cli import doctor as doctor_module

        monkeypatch.delenv("ESOREX_PLUGIN_DIR", raising=False)
        extra = Path("/extra/dir")
        result = doctor_module._candidate_recipe_dirs([extra])
        assert extra in result


class TestFindMatisseRecipeDir:
    """Unit tests for find_matisse_recipe_dir."""

    def test_returns_first_dir_with_matisse_recipes(self, monkeypatch, tmp_path):
        from matisse.cli import doctor as doctor_module

        good_dir = tmp_path / "plugins"
        good_dir.mkdir()

        monkeypatch.delenv("ESOREX_PLUGIN_DIR", raising=False)
        monkeypatch.setattr(
            doctor_module,
            "_list_esorex_recipes",
            lambda recipe_dir=None: ["mat_raw_estimates", "mat_cal_oifits"],
        )

        probe = doctor_module.find_matisse_recipe_dir(
            extra_candidates=[good_dir], verbose=False
        )
        assert probe is not None
        assert probe.matisse_recipes == ["mat_cal_oifits", "mat_raw_estimates"]

    def test_skips_dirs_without_matisse_recipes(self, monkeypatch, tmp_path):
        from matisse.cli import doctor as doctor_module

        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()

        monkeypatch.delenv("ESOREX_PLUGIN_DIR", raising=False)
        monkeypatch.setattr(
            doctor_module,
            "_list_esorex_recipes",
            lambda recipe_dir=None: ["other_recipe"],
        )

        probe = doctor_module.find_matisse_recipe_dir(
            extra_candidates=[empty_dir], verbose=False
        )
        assert probe is None

    def test_skips_failing_dirs(self, monkeypatch, tmp_path):
        from matisse.cli import doctor as doctor_module

        bad_dir = tmp_path / "bad"
        bad_dir.mkdir()

        monkeypatch.delenv("ESOREX_PLUGIN_DIR", raising=False)

        def _fail(recipe_dir=None):
            raise RuntimeError("esorex failed")

        monkeypatch.setattr(doctor_module, "_list_esorex_recipes", _fail)

        probe = doctor_module.find_matisse_recipe_dir(
            extra_candidates=[bad_dir], verbose=False
        )
        assert probe is None

    def test_returns_none_when_no_candidates(self, monkeypatch):
        from matisse.cli import doctor as doctor_module

        monkeypatch.delenv("ESOREX_PLUGIN_DIR", raising=False)
        # Override candidate dirs to return nothing (avoids probing real system paths)
        monkeypatch.setattr(
            doctor_module,
            "_candidate_recipe_dirs",
            lambda extra: [],
        )
        probe = doctor_module.find_matisse_recipe_dir(
            extra_candidates=[], verbose=False
        )
        assert probe is None


class TestCheckMatisseRecipes:
    """Unit tests for check_matisse_recipes."""

    def test_ok_with_matisse_recipes(self, monkeypatch):
        from matisse.cli import doctor as doctor_module

        monkeypatch.setattr(
            doctor_module,
            "_list_esorex_recipes",
            lambda recipe_dir=None: ["mat_raw_estimates", "other_recipe"],
        )

        result, recipes = doctor_module.check_matisse_recipes(
            recipe_dir=None, require_any=True
        )
        assert result.ok
        assert recipes == ["mat_raw_estimates"]

    def test_fail_require_any_no_mat_recipes(self, monkeypatch):
        from matisse.cli import doctor as doctor_module

        monkeypatch.setattr(
            doctor_module,
            "_list_esorex_recipes",
            lambda recipe_dir=None: ["other_recipe"],
        )

        result, recipes = doctor_module.check_matisse_recipes(
            recipe_dir=None, require_any=True
        )
        assert not result.ok
        assert result.fatal
        assert recipes == []

    def test_ok_no_require_any_and_no_mat_recipes(self, monkeypatch):
        from matisse.cli import doctor as doctor_module

        monkeypatch.setattr(
            doctor_module,
            "_list_esorex_recipes",
            lambda recipe_dir=None: ["other_recipe"],
        )

        result, recipes = doctor_module.check_matisse_recipes(
            recipe_dir=None, require_any=False
        )
        assert result.ok
        assert recipes == []

    def test_handles_esorex_failure(self, monkeypatch):
        from matisse.cli import doctor as doctor_module

        def _fail(recipe_dir=None):
            raise RuntimeError("broken")

        monkeypatch.setattr(doctor_module, "_list_esorex_recipes", _fail)

        result, recipes = doctor_module.check_matisse_recipes(
            recipe_dir=None, require_any=True
        )
        assert not result.ok
        assert result.fatal


class TestDoctorFullFlowMocked:
    """Integration-level tests for the full doctor command with mocked externals."""

    def _mock_esorex_found(self, monkeypatch, doctor_module):
        monkeypatch.setattr(
            doctor_module.shutil, "which", lambda cmd: "/usr/local/bin/esorex"
        )

    def test_esorex_found_recipes_found(self, monkeypatch):
        from matisse.cli import doctor as doctor_module

        self._mock_esorex_found(monkeypatch, doctor_module)
        monkeypatch.delenv("ESOREX_PLUGIN_DIR", raising=False)
        monkeypatch.setattr(
            doctor_module,
            "_list_esorex_recipes",
            lambda recipe_dir=None: ["mat_raw_estimates", "mat_cal_oifits"],
        )
        monkeypatch.setattr(
            doctor_module,
            "database_status",
            lambda: {"db.fits": "cached"},
        )

        result = runner.invoke(
            app,
            ["doctor", "--no-macports-probe"],
            catch_exceptions=False,
        )
        assert result.exit_code == 0
        assert "MATISSE environment looks OK" in result.output

    def test_esorex_found_no_recipes_require_any(self, monkeypatch):
        from matisse.cli import doctor as doctor_module

        self._mock_esorex_found(monkeypatch, doctor_module)
        monkeypatch.delenv("ESOREX_PLUGIN_DIR", raising=False)
        monkeypatch.setattr(
            doctor_module,
            "_list_esorex_recipes",
            lambda recipe_dir=None: ["other_recipe"],
        )
        monkeypatch.setattr(
            doctor_module,
            "database_status",
            lambda: {"db.fits": "cached"},
        )

        result = runner.invoke(
            app,
            ["doctor", "--no-macports-probe", "--require-any"],
            catch_exceptions=False,
        )
        assert result.exit_code == 2
        assert "no mat_* recipes found" in result.output

    def test_esorex_found_no_recipes_no_require_any(self, monkeypatch):
        from matisse.cli import doctor as doctor_module

        self._mock_esorex_found(monkeypatch, doctor_module)
        monkeypatch.delenv("ESOREX_PLUGIN_DIR", raising=False)
        monkeypatch.setattr(
            doctor_module,
            "_list_esorex_recipes",
            lambda recipe_dir=None: [],
        )
        monkeypatch.setattr(
            doctor_module,
            "database_status",
            lambda: {"db.fits": "cached"},
        )

        result = runner.invoke(
            app,
            ["doctor", "--no-macports-probe", "--no-require-any"],
            catch_exceptions=False,
        )
        assert result.exit_code == 0

    def test_esorex_found_with_env_plugin_dir(self, monkeypatch, tmp_path):
        from matisse.cli import doctor as doctor_module

        self._mock_esorex_found(monkeypatch, doctor_module)
        plugin_dir = tmp_path / "plugins"
        plugin_dir.mkdir()
        monkeypatch.setenv("ESOREX_PLUGIN_DIR", str(plugin_dir))
        monkeypatch.setattr(
            doctor_module,
            "_list_esorex_recipes",
            lambda recipe_dir=None: ["mat_raw_estimates"],
        )
        monkeypatch.setattr(
            doctor_module,
            "database_status",
            lambda: {"db.fits": "cached"},
        )

        result = runner.invoke(
            app,
            ["doctor", "--no-macports-probe"],
            catch_exceptions=False,
        )
        assert result.exit_code == 0
        assert "ESOREX_PLUGIN_DIR" in result.output

    def test_calibrator_db_all_cached(self, monkeypatch):
        from matisse.cli import doctor as doctor_module

        self._mock_esorex_found(monkeypatch, doctor_module)
        monkeypatch.delenv("ESOREX_PLUGIN_DIR", raising=False)
        monkeypatch.setattr(
            doctor_module,
            "_list_esorex_recipes",
            lambda recipe_dir=None: ["mat_raw_estimates"],
        )
        monkeypatch.setattr(
            doctor_module,
            "database_status",
            lambda: {"a.fits": "cached", "b.fits": "cached"},
        )

        result = runner.invoke(
            app,
            ["doctor", "--no-macports-probe"],
            catch_exceptions=False,
        )
        assert result.exit_code == 0
        assert "2/2 available" in result.output

    def test_forced_recipe_dir(self, monkeypatch, tmp_path):
        from matisse.cli import doctor as doctor_module

        self._mock_esorex_found(monkeypatch, doctor_module)
        monkeypatch.delenv("ESOREX_PLUGIN_DIR", raising=False)

        forced_dir = tmp_path / "my_recipes"
        forced_dir.mkdir()

        called_with: list[Path | None] = []

        def _fake_list(recipe_dir=None):
            called_with.append(recipe_dir)
            return ["mat_raw_estimates"]

        monkeypatch.setattr(doctor_module, "_list_esorex_recipes", _fake_list)
        monkeypatch.setattr(
            doctor_module,
            "database_status",
            lambda: {"db.fits": "cached"},
        )

        result = runner.invoke(
            app,
            ["doctor", "--recipe-dir", str(forced_dir)],
            catch_exceptions=False,
        )
        assert result.exit_code == 0
        assert f"forced: {forced_dir}" in result.output
        # check_matisse_recipes should have been called with the forced dir
        assert forced_dir in called_with
