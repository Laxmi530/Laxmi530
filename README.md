- 👋 Hi, I’m Laxmidhar
- 👀 I’m interested in writing code.
- 🌱 I’m currently learning AI, ML, Data science and computer vision
- 💞️ I’m looking to collaborate on AI, ML projects...
- 📫 How to reach me...
- ✉️ laxmidhar530@outlook.com

- Code
  
```
"""
Check that the app's Python packages are installed, and install any that are
missing, correctly.

Standard library only: this has to work before anything else is installed.

How installing works
--------------------
Every missing package installs in one ``pip install`` under a constraints file
holding every pin in ``requirements.txt``, so no dependency can move a tested
version.

A package whose declared requirements contradict the tested pins can be listed
in :data:`NO_DEPS`. It is then installed with ``--no-deps``, followed by its
own declared dependencies minus the pinned ones. That second phase existed for
``prophet==1.2.2``, which declared ``numpy<2.4`` and ``pandas<3`` against
numpy 2.4.6 and pandas 3.0.3, so a plain ``pip install -r requirements.txt``
failed with ``ResolutionImpossible``. Prophet 1.3.0 dropped those caps, the pin
is now 1.5.0, and the list is empty (README section 76). The mechanism stays
for the next package that over-constrains a pin.

Entry points: ``python run_app.py`` (check, install, launch) and the guard at
the top of ``UI.py`` (an install page instead of a ``ModuleNotFoundError``).
Installed packages at a different version from the pin are reported, never
changed: replacing someone's working package is not this module's call.
"""

import os
import re
import subprocess
import sys
import sysconfig
import tempfile
from importlib import metadata

ROOT = os.path.dirname(os.path.abspath(__file__))
REQUIREMENTS = os.path.join(ROOT, "requirements.txt")
#: Packages whose declared requirements contradict the tested pins. Empty since
#: prophet 1.5.0 (1.2.2 capped numpy<2.4 and pandas<3).
NO_DEPS = set()
MIN_PYTHON = (3, 10)

_LINE = re.compile(r"^\s*([A-Za-z0-9][A-Za-z0-9._-]*)\s*(\[[^\]]*\])?\s*(.*?)\s*$")


def _norm(name):
    return re.sub(r"[-_.]+", "-", name).lower()


def read_requirements(path=REQUIREMENTS):
    """
    ``[(name, spec, pinned_version_or_None)]`` from a requirements file.

    Comments, blank lines and options (``-r``, ``--index-url``) are skipped;
    ``spec`` is the original requirement string for pip.
    """
    out = []
    with open(path, encoding="utf-8") as f:
        for raw in f:
            line = raw.split("#", 1)[0].strip()
            if not line or line.startswith("-"):
                continue
            m = _LINE.match(line)
            if not m:
                continue
            name, _extras, rest = m.groups()
            pin = rest[2:].strip() if rest.startswith("==") else None
            out.append((name, line, pin))
    return out


def installed_version(name):
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def check(reqs=None):
    """``(missing, mismatched)``: lists of ``(name, spec, pin, installed)``."""
    reqs = read_requirements() if reqs is None else reqs
    missing, mismatched = [], []
    for name, spec, pin in reqs:
        have = installed_version(name)
        if have is None:
            missing.append((name, spec, pin, None))
        elif pin and have != pin:
            mismatched.append((name, spec, pin, have))
    return missing, mismatched


def missing(reqs=None):
    """The requirements that are not installed at all (names)."""
    return [m[0] for m in check(reqs)[0]]


def _needs_user_flag():
    """Install with --user when outside a virtualenv and site-packages is read-only."""
    in_venv = sys.prefix != getattr(sys, "base_prefix", sys.prefix)
    return not in_venv and not os.access(sysconfig.get_paths()["purelib"], os.W_OK)


def _declared_deps(name, skip):
    """A package's installed, unconditional requirements, minus names in ``skip``."""
    out = []
    for r in metadata.requires(name) or []:
        if ";" in r and "extra" in r.split(";", 1)[1]:
            continue                      # optional extras ("dev", "parallel")
        dep = _LINE.match(r.split(";", 1)[0])
        if dep and _norm(dep.group(1)) not in skip:
            out.append(r.split(";", 1)[0].strip())
    return out


def plan(missing_specs, reqs):
    """
    The pip commands, in order, for the missing requirements.

    Returns a list of argument lists (without the ``python -m pip`` prefix);
    a ``("DEPS_OF", name)`` tuple marks where a NO_DEPS package's own
    dependencies are installed, which can only be read once it is installed.
    """
    no_deps = {_norm(n) for n in NO_DEPS}
    normal = [s for n, s, _ in missing_specs if _norm(n) not in no_deps]
    special = [(n, s) for n, s, _ in missing_specs if _norm(n) in no_deps]
    steps = []
    if normal:
        steps.append(["install", "-c", "{constraints}", *normal])
    for n, s in special:
        steps.append(["install", "--no-deps", s])
        steps.append(("DEPS_OF", n))
    return steps


def install(missing_specs=None, reqs=None, runner=None, log=print):
    """
    Install the missing requirements. Returns True when nothing is missing
    afterwards. ``runner(cmd) -> returncode`` is injectable for tests.
    """
    reqs = read_requirements() if reqs is None else reqs
    if missing_specs is None:
        missing_specs = [(n, s, p) for n, s, p, _ in check(reqs)[0]]
    if not missing_specs:
        return True
    if runner is None:
        def runner(cmd):
            log("$ " + " ".join(cmd))
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                    text=True, encoding="utf-8", errors="replace")
            for line in proc.stdout:
                log(line.rstrip())
            return proc.wait()
    pinned = {_norm(n) for n, _, p in reqs if p}
    no_deps = {_norm(n) for n in NO_DEPS}
    with tempfile.NamedTemporaryFile("w", suffix="-constraints.txt", delete=False,
                                     encoding="utf-8") as f:
        for n, s, p in reqs:
            if p and _norm(n) not in no_deps:
                f.write(f"{n}=={p}\n")
        constraints = f.name
    base = [sys.executable, "-m", "pip"]
    extra = ["--user"] if _needs_user_flag() else []
    try:
        for step in plan(missing_specs, reqs):
            if isinstance(step, tuple):
                deps = _declared_deps(step[1], skip=pinned)
                if not deps:
                    continue
                cmd = base + ["install", *extra, "-c", constraints, *deps]
            else:
                cmd = base + [step[0], *extra] + [constraints if a == "{constraints}" else a
                                                  for a in step[1:]]
            if runner(cmd) != 0:
                log("pip reported an error; stopping. Fix it and run again: "
                    "nothing that installed is lost.")
                return False
            if isinstance(step, tuple):
                log(f"Note: pip may print 'ERROR: ... dependency conflicts' saying {step[1]} "
                    "requires an older version of a pinned package. That is expected: the "
                    "declaration is conservative, and the app was verified on the pinned "
                    "versions (see requirements.txt).")
    finally:
        os.unlink(constraints)
    import importlib
    importlib.invalidate_caches()
    left = missing(reqs)
    if left:
        log("Still missing after install: " + ", ".join(left))
    return not left


def python_ok():
    return sys.version_info[:2] >= MIN_PYTHON


def render_install_page(st, missing_names):
    """
    A Streamlit page listing what is missing, with a button that installs it.
    Called from the top of UI.py before any other import, so it can only use
    Streamlit itself. Always ends with ``st.stop()``.
    """
    st.set_page_config(page_title="Forecasting App - setup", page_icon="📦")
    st.title("Some required packages are not installed")
    st.write(f"{len(missing_names)} of the packages in `requirements.txt` are missing, so the "
             "app cannot start yet:")
    st.code("\n".join(missing_names))
    st.write("Installing downloads them from PyPI into this Python environment "
             f"(`{sys.executable}`). The deep-learning packages are large, so a first install "
             "can take several minutes and a few GB.")
    if st.button("Install missing packages", type="primary"):
        lines = []
        box = st.empty()

        def log(line):
            lines.append(line)
            box.code("\n".join(lines[-40:]))
        with st.spinner("Installing..."):
            ok = install(log=log)
        if ok:
            st.success("Everything is installed. Reloading the app...")
            st.rerun()
        else:
            st.error("The install did not finish; the log above shows why. You can also run "
                     "`python run_app.py` from a terminal.")
    st.caption("Or, from a terminal in the project folder: `python run_app.py` (checks, "
               "installs what is missing, then starts the app).")
    st.stop()
```
