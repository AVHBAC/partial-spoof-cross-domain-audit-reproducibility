"""Run all verification scripts and report a summary.

A script PASSES iff it exits 0 AND prints `ERRORS: 0`. Any non-zero exit
(crash/exception) or a positive `ERRORS:` count is a FAILURE. The runner
itself exits non-zero if any script fails, so `python run_all.py; echo $?`
is meaningful for CI / one-command reproduction.
"""
import re
import subprocess
import sys
from pathlib import Path

# verify_metadata.py is intentionally NOT in this list: it validates utterance
# counts against the raw ASVspoof/PartialSpoof protocol files, which are external
# downloads not bundled in this package. Run it separately if those are present.
SCRIPTS = [
    "verify_e1_baseline.py",
    "verify_e5_cross_dataset.py",
    "verify_voting.py",
    "verify_hqmpsd.py",
    "verify_partialedit.py",
    "verify_utt_confusion.py",
]

_ERRORS_RE = re.compile(r"^ERRORS:\s*(\d+)\s*$", re.MULTILINE)


def _error_count(stdout: str):
    """Return the integer reported on the script's `ERRORS: N` line.

    Returns None if the script printed no such line (treated as a failure,
    since every verify script is expected to emit one)."""
    matches = _ERRORS_RE.findall(stdout)
    if not matches:
        return None
    # If a script prints multiple ERRORS: lines, the run fails unless all are 0.
    return sum(int(m) for m in matches)


def main():
    script_dir = Path(__file__).parent
    results = {}

    for script in SCRIPTS:
        print(f"\n{'#'*70}")
        print(f"  RUNNING: {script}")
        print(f"{'#'*70}")
        r = subprocess.run(
            [sys.executable, str(script_dir / script)],
            capture_output=True,
            text=True,
        )
        # Stream the captured output so the run is still human-readable.
        if r.stdout:
            print(r.stdout, end="" if r.stdout.endswith("\n") else "\n")
        if r.stderr:
            print(r.stderr, end="" if r.stderr.endswith("\n") else "\n")
        results[script] = (r.returncode, _error_count(r.stdout))

    print(f"\n\n{'='*70}")
    print("VERIFICATION SUMMARY")
    print(f"{'='*70}")
    all_pass = True
    for script, (rc, n_err) in results.items():
        if rc != 0:
            status, detail = "FAIL", f"exit={rc}"
        elif n_err is None:
            status, detail = "FAIL", "no ERRORS: line"
        elif n_err > 0:
            status, detail = "FAIL", f"{n_err} error(s)"
        else:
            status, detail = "PASS", "0 errors"
        if status == "FAIL":
            all_pass = False
        print(f"  {script:<35s} {status:<5s} ({detail})")

    print(f"\n  Overall: {'ALL PASSED' if all_pass else 'SOME FAILED'}")
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
