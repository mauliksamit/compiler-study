"""CPU check of the ALGORITHM used by notebook 2's looped Triton kernel.

This emulates, in NumPy, the per-lane online-softmax recurrence of
`softmax_row_looped_kernel` (running max m and rescaled running sum s per
lane, combined after the loop, with the -inf guards) and compares it with a
float64 softmax. It verifies the math and the guards, NOT the Triton kernel
itself (Triton does not run on this Windows machine; the kernel's own
correctness gate is in the notebook and is NOT RUN until the notebook runs).
"""
import numpy as np
import pytest


def looped_softmax_emulation(row: np.ndarray, block: int) -> np.ndarray:
    n = row.shape[0]
    m_vec = np.full(block, -np.inf, dtype=np.float64)
    s_vec = np.zeros(block, dtype=np.float64)
    for start in range(0, n, block):
        x = np.full(block, -np.inf)
        seg = row[start:start + block]
        x[: seg.shape[0]] = seg
        m_new = np.maximum(m_vec, x)
        with np.errstate(invalid="ignore"):
            alpha = np.where(m_vec == m_new, 1.0, np.exp(m_vec - m_new))
            term = np.where(x == -np.inf, 0.0, np.exp(x - m_new))
        s_vec = s_vec * alpha + term
        m_vec = m_new
    row_max = m_vec.max()
    with np.errstate(invalid="ignore"):
        scale = np.where(m_vec == -np.inf, 0.0, np.exp(m_vec - row_max))
    row_sum = (s_vec * scale).sum()
    return np.exp(row - row_max) / row_sum


def ref_softmax(row: np.ndarray) -> np.ndarray:
    e = np.exp(row - row.max())
    return e / e.sum()


@pytest.mark.parametrize("n", [1, 3, 4096, 4097, 8193, 30001])
@pytest.mark.parametrize("kind", ["randn", "large", "neg_inf_lanes", "neg_inf_block"])
def test_matches_reference(n, kind):
    rng = np.random.default_rng(n)
    row = rng.standard_normal(n)
    if kind == "large":
        row = row * 80
    if kind == "neg_inf_lanes" and n > 1:
        row[::3] = -np.inf
    if kind == "neg_inf_block" and n > 4096:
        row[:4096] = -np.inf  # an entire first block of -inf: the NaN guard case
    out = looped_softmax_emulation(row, block=4096)
    assert np.all(np.isfinite(out))
    np.testing.assert_allclose(out, ref_softmax(row), rtol=1e-12, atol=1e-15)


def test_guard_matters():
    """Without the guards an all -inf first block yields NaN (why the guards exist)."""
    row = np.concatenate([np.full(4096, -np.inf), np.zeros(10)])
    m = np.full(4096, -np.inf)
    x = row[:4096]
    with np.errstate(invalid="ignore"):
        unguarded = np.exp(x - np.maximum(m, x))
    assert np.isnan(unguarded).all()
