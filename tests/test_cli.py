import subprocess
import sys

from mc_pricer.cli import main


def test_quick_run_prints_every_section(capsys):
    assert main(["--quick", "--hedge-steps", "12", "52"]) == 0
    out = capsys.readouterr().out
    for heading in ("pricing", "Greeks", "Convergence", "Variance reduction", "delta hedging"):
        assert heading in out


def test_put_and_section_selection(capsys):
    assert main(["--quick", "--option-type", "put", "--method", "control_variate", "--sections", "pricing"]) == 0
    out = capsys.readouterr().out
    assert "European put" in out and "Greeks" not in out


def test_invalid_input_returns_exit_code_2(capsys):
    assert main(["--sigma", "-0.2", "--sections", "pricing"]) == 2
    assert "sigma must be non-negative" in capsys.readouterr().err


def test_module_entry_point():
    proc = subprocess.run(
        [sys.executable, "-m", "mc_pricer", "--quick", "--sections", "pricing"],
        capture_output=True, text=True, check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert "Black-Scholes price" in proc.stdout
