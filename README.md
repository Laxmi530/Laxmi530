- 👋 Hi, I’m Laxmidhar
- 👀 I’m interested in writing code.
- 🌱 I’m currently learning AI, ML, Data science and computer vision
- 💞️ I’m looking to collaborate on AI, ML projects...
- 📫 How to reach me...
- ✉️ laxmidhar530@outlook.com

- Code
  
```
"""
Declarative neural-network architectures, and the record of what was built.

Why this module exists
----------------------
This project ships an auditable per-run manifest (``reproducibility.run_manifest``)
carrying the seed, package versions and request parameters — and it did **not**
carry the network shape. That is a hole in a reproducibility claim: pinning the
seed reproduces a fit only if the architecture is also fixed, and the layer
stacks lived as literals inside ``DL_model.py``, ``CNN_LSTM_model.py`` and
``vol_lstm_model.py``. Edit a hidden size, and every prior manifest silently
describes a model that no longer exists while still claiming to be complete.

Two mechanisms, with deliberately different jobs
------------------------------------------------
1. :data:`SPECS` — the **declared** architecture per model, JSON-serialisable
   and overridable from a file. These genuinely drive the builders that accept
   them, so changing a spec changes the network without touching model code.
2. :func:`describe` — the **realised** architecture, read back off the compiled
   Keras object: layer types, output shapes, parameter counts, optimizer, loss.

The manifest records the realised one, and that choice matters. A declared spec
that is merely *stored* alongside a run is decoration — the same "configuration
theatre" ``config.py``'s tests exist to prevent. Introspection cannot drift from
what actually ran, and it covers models whose builders are not spec-driven,
which is why it is the authoritative record rather than the convenient one.

Honest scope
------------
Spec-driven today: **DL (LSTM/GRU)**, **CNN-LSTM**, **Vol-LSTM** — the three whose
builders take their layer sizes as arguments. N-BEATS, TCN and the Transformer
are **introspection-only**: their shapes are recorded faithfully but not yet
controlled from here. Wiring those is mechanical follow-up work, and the
distinction is reported by :func:`spec_driven` rather than glossed over.

Overriding
----------
Drop an ``architectures.json`` next to this file (or point ``STOCKAPP_ARCH_FILE``
at one) containing any subset of :data:`SPECS`; matching keys are merged over the
defaults. An override is reported in the console banner and in the manifest,
because a run on a non-default architecture is not the shipped configuration.
"""

import copy
import hashlib
import json
import os

import config as _cfg

# Models whose builders read their layer sizes from SPECS. Everything else is
# recorded by introspection only — see the module docstring.
SPEC_DRIVEN = ("DL", "CNN-LSTM", "Vol-LSTM")

ARCH_FILE_ENV = _cfg.ENV_PREFIX + "ARCH_FILE"
DEFAULT_ARCH_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 "architectures.json")

# -----------------------------------------------------------------------------
# Declared architectures
# -----------------------------------------------------------------------------
# Values match the literals these models shipped with, so introducing this module
# changes no forecast. Verified by tests/test_architectures.py, which asserts the
# defaults reproduce the previous layer stacks exactly.
_DEFAULT_SPECS = {
    "DL": {
        "description": "Stacked LSTM/GRU, direct multi-horizon (DL_model.py)",
        "recurrent_units": [64, 64, 32],
        "recurrent_dropout": 0.1,
        "dropout": [0.3, 0.3, 0.2],
        "dense_units": 16,
        "dense_activation": "relu",
    },
    "CNN-LSTM": {
        "description": "Causal Conv1D -> pool -> stacked LSTM (CNN_LSTM_model.py)",
        "filters": [32, 64],
        "kernel_size": 3,
        "pool_size": 2,
        "lstm_units": [48, 24],
        "dropout": 0.2,
    },
    "Vol-LSTM": {
        "description": "Stacked LSTM over RV channels (vol_lstm_model.py)",
        "units": [48, 24],
        "dropout": 0.2,
    },
}


def _load_overrides():
    """Read the optional JSON override file; never raise on a bad file."""
    path = os.environ.get(ARCH_FILE_ENV) or DEFAULT_ARCH_FILE
    if not os.path.exists(path):
        return {}, None
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return (data if isinstance(data, dict) else {}), path
    except Exception:
        # A malformed architecture file must not stop the app; it should simply
        # fail to take effect, exactly as a malformed feature flag does.
        return {}, None


def _build_specs():
    specs = copy.deepcopy(_DEFAULT_SPECS)
    overrides, path = _load_overrides()
    applied = []
    for name, patch in overrides.items():
        if name in specs and isinstance(patch, dict):
            specs[name].update(patch)
            applied.append(name)
    return specs, sorted(applied), path


SPECS, OVERRIDDEN, ARCH_FILE = _build_specs()


def spec(name):
    """Return a copy of one model's declared spec ({} if unknown)."""
    return copy.deepcopy(SPECS.get(name, {}))


def spec_driven(name):
    """Whether ``name``'s builder actually reads its shape from :data:`SPECS`."""
    return name in SPEC_DRIVEN


def fingerprint(obj):
    """
    Short, stable hash of any JSON-serialisable object.

    Lets the manifest carry an architecture identity in one short token, so two
    runs can be compared for "same network?" without diffing nested dicts.
    Sorted keys make it order-independent.
    """
    try:
        blob = json.dumps(obj, sort_keys=True, default=str)
    except Exception:
        blob = repr(obj)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:12]


# -----------------------------------------------------------------------------
# Realised architecture (introspection)
# -----------------------------------------------------------------------------
def describe(model, name=None):
    """
    Read the architecture back off a compiled Keras model.

    Returns a small JSON-safe dict — never the full Keras config, which is large
    and version-specific. Records what was *built*, so it cannot disagree with
    the run it describes.

    Every failure path returns a dict with ``available=False``: a manifest is an
    audit record, and failing to describe a model must not break the forecast it
    describes.
    """
    out = {"name": name, "available": False}
    if model is None:
        return out
    try:
        layers = []
        for layer in getattr(model, "layers", []):
            entry = {"type": type(layer).__name__,
                     "name": getattr(layer, "name", None)}
            try:
                shape = layer.output.shape
                entry["output_shape"] = [None if d is None else int(d)
                                         for d in shape]
            except Exception:
                pass
            for attr in ("units", "filters", "rate", "kernel_size",
                         "pool_size", "return_sequences", "activation"):
                if hasattr(layer, attr):
                    val = getattr(layer, attr)
                    val = getattr(val, "__name__", val)
                    if isinstance(val, (int, float, bool, str)):
                        entry[attr] = val
                    elif isinstance(val, (list, tuple)):
                        entry[attr] = list(val)
            layers.append(entry)

        out.update({
            "available": True,
            "layers": layers,
            "n_layers": len(layers),
            "n_params": int(model.count_params()),
            "loss": _readable(getattr(model, "loss", None)),
            "optimizer": type(getattr(model, "optimizer", None)).__name__,
        })
        if name and name in SPECS:
            out["declared_spec"] = spec(name)
            out["spec_driven"] = spec_driven(name)
        out["fingerprint"] = fingerprint(
            {"layers": layers, "loss": out["loss"], "optimizer": out["optimizer"]})
    except Exception as exc:
        out["note"] = f"introspection failed: {str(exc)[:120]}"
    return out


def _readable(obj):
    if obj is None:
        return None
    return getattr(obj, "__name__", None) or getattr(
        obj, "name", None) or str(obj)


def attach(prediction_df, model, name):
    """
    Record a model's realised architecture on its prediction frame.

    Uses ``DataFrame.attrs``, the convention this project already uses to carry
    ``risk_log_returns`` from a model to the UI. Best-effort and silent: an audit
    annotation must never be able to fail a forecast.
    """
    try:
        prediction_df.attrs["architecture"] = describe(model, name)
    except Exception:
        pass
    return prediction_df


def manifest_block(architectures):
    """
    Condense per-model architecture records for the run manifest.

    Stores the compact form (layer count, parameter count, fingerprint) rather
    than every layer: the manifest is read by humans in an expander, and a
    six-model layer dump would bury the seed and versions that sit next to it.
    The fingerprint is what makes "was this the same network?" answerable.
    """
    block = {}
    for name, arch in (architectures or {}).items():
        if not isinstance(arch, dict) or not arch.get("available"):
            continue
        block[name] = {
            "n_layers": arch.get("n_layers"),
            "n_params": arch.get("n_params"),
            "optimizer": arch.get("optimizer"),
            "loss": arch.get("loss"),
            "fingerprint": arch.get("fingerprint"),
            "spec_driven": arch.get("spec_driven", False),
        }
    return block


def describe_specs():
    """Plain-text dump of every declared spec (for ``python architectures.py``)."""
    lines = []
    for name, sp in SPECS.items():
        mark = "spec-driven" if spec_driven(name) else "introspection-only"
        flag = "  [OVERRIDDEN]" if name in OVERRIDDEN else ""
        lines.append(f"{name}  ({mark}){flag}")
        for k, v in sp.items():
            lines.append(f"    {k}: {v}")
        lines.append(f"    fingerprint: {fingerprint(sp)}")
        lines.append("")
    if ARCH_FILE and OVERRIDDEN:
        lines.append(f"overrides loaded from: {ARCH_FILE}")
    lines.append("Models not listed (N-BEATS, TCN, Transformer) are recorded by "
                 "introspection but not yet driven from here.")
    return "\n".join(lines)


if __name__ == "__main__":
    print(describe_specs())

```
