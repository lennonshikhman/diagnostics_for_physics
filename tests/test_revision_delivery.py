import json

import pytest
from test_revision_analysis_exports import load_script

from diagnostic_suite import __version__


def test_release_counts_require_a_complete_passing_run():
    module = load_script("generate_revision_delivery")
    assert module.pytest_count("97 passed in 3.25s\n") == 97
    assert module.pytest_count("98 passed, 1 warning in 4.0s\n") == 98
    with pytest.raises(ValueError, match="complete passing"):
        module.pytest_count("96 passed, 1 failed in 4.0s")
    with pytest.raises(ValueError, match="complete passing"):
        module.pytest_count("...still running...")


def test_response_locations_follow_compiled_renumbering_and_appendix_labels():
    module = load_script("generate_revision_delivery")
    aux = (
        r"\newlabel{sec:rollout}{{10.4}{31}{Rollout}{subsection.10.4}{}}"
        + "\n"
        + r"\newlabel{tab:rollout}{{9}{32}{Results}{table.9}{}}"
        + "\n"
        + r"\newlabel{app:contracts}{{Appendix A}{41}{}{appendix.A}{}}"
    )
    values = module.compiled_locations(aux)
    assert values["sec:rollout"] == "Section~10.4 (p.~31)"
    assert values["tab:rollout"] == "Table~9 (p.~32)"
    assert values["app:contracts"] == "Appendix~A (p.~41)"


def test_release_claim_rejects_a_missing_installed_balance_check(tmp_path):
    module = load_script("generate_revision_delivery")
    result = {
        "passed": True,
        "resume_preserved_checkpoint_bytes": True,
        "installation": {"version": __version__},
        "commands": [
            {"name": key, "exit_code": 0}
            for key in ["pytest", "self_check", "referee", "sensitivity", "training", "resume"]
        ],
    }
    (tmp_path / "validation.json").write_text(json.dumps(result))
    (tmp_path / "pytest.log").write_text("100 passed in 4s\n")
    with pytest.raises(ValueError, match="command is missing"):
        module.validated_installation(tmp_path, 100)
    result["commands"].append({"name": "balance_constraints", "exit_code": 0})
    (tmp_path / "validation.json").write_text(json.dumps(result))
    assert module.validated_installation(tmp_path, 100)["version"] == __version__
    installation = dict(
        version=__version__,
        python="3.10",
        torch="2.0",
        numpy="1.23",
        pandas="1.5",
        h5py="3.8",
        source_sha256="verified",
    )
    manifest = dict(
        package_version=__version__,
        source_sha256="verified",
        checks_passed=16,
        checks_total=16,
        torch_threads=1,
        elapsed_seconds=0.0283,
    )
    text = module.environment_statement(installation, manifest, 100)
    assert "All 100 tests" in text and "0.028 seconds" in text and __version__ in text
    manifest["package_version"] = "0.3.0"
    with pytest.raises(ValueError, match="different release/source"):
        module.environment_statement(installation, manifest, 100)


def test_pdf_equivalence_ignores_word_hyphenation_but_preserves_numeric_and_algebraic_signs():
    module = load_script("check_revision_pdfs")
    first = "kinetic contri- bution; Full-domain; en- vironments; x-y; 1-2"
    second = "kinetic contribution; Full- domain; environ- ments; x-y; 1-2"
    assert module.equivalent_text(first) == module.equivalent_text(second)
    assert module.equivalent_text(first) != module.equivalent_text(second.replace("1-2", "1+2"))
    assert "x-y" in module.equivalent_text(first)
    colors = {"(0.0,)": 300, "(0.0, 0.27452, 0.66667)": 1500}
    revised = {"color_counts": colors, "pages": [{"colors": colors}, {"colors": colors}, {"colors": colors}]}
    # A newly revised introduction is blue; protected identity text stays black.
    module.validate_revision_colors(revised)
    revised["pages"][0] = {"colors": {"(0.0, 0.27452, 0.66667)": 1800}}
    with pytest.raises(ValueError, match="protected"):
        module.validate_revision_colors(revised)
