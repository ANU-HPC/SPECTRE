# SPDX-FileCopyrightText: 2026 Graz University of Technology
#
# SPDX-License-Identifier: MIT

"""Check that postprocessing writes the accepted interface state."""

from pathlib import Path

import numpy as np

from spectre import Force_Minimizer, SPECTREout, input_dict


def interfaces(out):
    nvol = out.input.physics.Nvol
    names = ("Rbc", "Zbs", "Rbs", "Zbc")
    return np.concatenate([getattr(out.output, name)[1:nvol].ravel() for name in names])


def test_postprocess_writes_accepted_state_after_rejected_trial(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    def reject_last_trial(obj, x_init, f, jac):
        trial = x_init.copy()
        trial[0] += 1e-3
        f(trial, obj)
        return input_dict({"x": x_init, "jac": None})

    source = str(Path(__file__).with_name("qa_3vol_light.toml"))
    obj = Force_Minimizer(source, optimizer=reject_last_trial, save_x_iter=False)
    accepted = interfaces(SPECTREout.from_spectre(obj.field_obj))
    obj.minimize()
    obj.postprocess(save_json=False, save_pkl=False)
    written = interfaces(SPECTREout(f"{obj.field_obj.name}_opt.h5"))
    np.testing.assert_array_equal(written, accepted)
