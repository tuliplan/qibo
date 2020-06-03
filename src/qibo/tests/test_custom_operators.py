"""
Testing Tensorflow custom operators circuit.
"""
import pytest
import numpy as np
import tensorflow as tf
from qibo.tensorflow import custom_operators as op

_atol = 1e-6


@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
@pytest.mark.parametrize("compile", [False, True])
def test_initial_state(dtype, compile):
  """Check that initial_state updates first element properly."""
  def apply_operator(dtype):
    """Apply the initial_state operator"""
    a = tf.zeros(10, dtype=dtype)
    return op.initial_state(a)

  func = apply_operator
  if compile:
      func = tf.function(apply_operator)
  final_state = func(dtype)
  exact_state = np.array([1] + [0]*9, dtype=dtype)
  np.testing.assert_allclose(final_state, exact_state)


def tensorflow_random_complex(shape, dtype):
  _re = tf.random.uniform(shape, dtype=dtype)
  _im = tf.random.uniform(shape, dtype=dtype)
  return tf.complex(_re, _im)


@pytest.mark.parametrize(("nqubits", "target", "dtype", "compile", "einsum_str"),
                         [(5, 4, np.float32, False, "abcde,Ee->abcdE"),
                          (4, 2, np.float32, True, "abcd,Cc->abCd"),
                          (4, 2, np.float64, False, "abcd,Cc->abCd"),
                          (3, 0, np.float64, True, "abc,Aa->Abc"),
                          (8, 5, np.float64, False, "abcdefgh,Ff->abcdeFgh")])
def test_apply_gate(nqubits, target, dtype, compile, einsum_str):
    """Check that ``op.apply_gate`` agrees with ``tf.einsum``."""
    def apply_operator(state, gate):
      return op.apply_gate(state, gate, nqubits, target)

    state = tensorflow_random_complex((2 ** nqubits,), dtype)
    gate = tensorflow_random_complex((2, 2), dtype)

    target_state = tf.reshape(state, nqubits * (2,))
    target_state = tf.einsum(einsum_str, target_state, gate)
    target_state = target_state.numpy().ravel()

    if compile:
        apply_operator = tf.function(apply_operator)
    state = apply_operator(state, gate)
    np.testing.assert_allclose(target_state, state.numpy(), atol=_atol)


@pytest.mark.parametrize(("nqubits", "compile"),
                         [(2, True), (3, False), (4, True), (5, False)])
def test_apply_gate_cx(nqubits, compile):
    """Check ``op.apply_gate`` for multiply-controlled X gates."""
    state = tensorflow_random_complex((2 ** nqubits,), dtype=tf.float64)

    target_state = state.numpy()
    gate = np.eye(2 ** nqubits, dtype=target_state.dtype)
    gate[-2, -2], gate[-2, -1] = 0, 1
    gate[-1, -2], gate[-1, -1] = 1, 0
    target_state = gate.dot(target_state)

    xgate = tf.cast([[0, 1], [1, 0]], dtype=state.dtype)
    controls = list(range(nqubits - 1))
    def apply_operator(state):
      return op.apply_gate(state, xgate, nqubits, nqubits - 1, controls)
    if compile:
        apply_operator = tf.function(apply_operator)
    state = apply_operator(state)

    np.testing.assert_allclose(target_state, state.numpy())


@pytest.mark.parametrize(("nqubits", "target", "controls", "compile", "einsum_str"),
                         [(3, 0, [1, 2], False, "a,Aa->A"),
                          (4, 3, [0, 1, 2], True, "a,Aa->A"),
                          (5, 3, [1], True, "abcd,Cc->abCd"),
                          (5, 2, [1, 4], True, "abc,Bb->aBc"),
                          (6, 3, [0, 2, 5], False, "abc,Bb->aBc"),
                          (6, 3, [0, 2, 4, 5], False, "ab,Bb->aB")])
def test_apply_gate_controlled(nqubits, target, controls, compile, einsum_str):
    """Check ``op.apply_gate`` for random controlled gates."""
    state = tensorflow_random_complex((2 ** nqubits,), dtype=tf.float64)
    gate = tensorflow_random_complex((2, 2), dtype=tf.float64)

    target_state = state.numpy().reshape(nqubits * (2,))
    slicer = nqubits * [slice(None)]
    for c in controls:
        slicer[c] = 1
    slicer = tuple(slicer)
    target_state[slicer] = np.einsum(einsum_str, target_state[slicer], gate)
    target_state = target_state.ravel()

    def apply_operator(state):
      return op.apply_gate(state, gate, nqubits, target, controls)
    if compile:
        apply_operator = tf.function(apply_operator)

    state = apply_operator(state)
    np.testing.assert_allclose(target_state, state.numpy())


@pytest.mark.parametrize("compile", [False, True])
def test_apply_gate_error(compile):
    """Check that ``TypeError`` is raised for invalid ``controls``."""
    state = tensorflow_random_complex((2 ** 2,), dtype=tf.float64)
    gate = tensorflow_random_complex((2, 2), dtype=tf.float64)

    def apply_operator(state):
      return op.apply_gate(state, gate, 2, 0, "a")
    if compile:
        apply_operator = tf.function(apply_operator)
    with pytest.raises(TypeError):
        state = apply_operator(state)


@pytest.mark.parametrize(("nqubits", "target", "gate"),
                         [(3, 0, "x"), (4, 3, "x"),
                          (5, 2, "y"), (3, 1, "z")])
@pytest.mark.parametrize("compile", [False, True])
def test_apply_pauli_gate(nqubits, target, gate, compile):
    """Check ``apply_x``, ``apply_y`` and ``apply_z`` kernels."""
    matrices = {"x": np.array([[0, 1], [1, 0]], dtype=np.complex128),
                "y": np.array([[0, -1j], [1j, 0]], dtype=np.complex128),
                "z": np.array([[1, 0], [0, -1]], dtype=np.complex128)}
    state = tensorflow_random_complex((2 ** nqubits,), dtype=tf.float64)
    target_state = tf.cast(state.numpy(), dtype=state.dtype)
    target_state = op.apply_gate(state, matrices[gate], nqubits, target)

    def apply_operator(state):
      return getattr(op, "apply_{}".format(gate))(state, nqubits, target)
    if compile:
        apply_operator = tf.function(apply_operator)
    state = apply_operator(state)

    np.testing.assert_allclose(target_state.numpy(), state.numpy())


@pytest.mark.parametrize(("nqubits", "target", "controls"),
                         [(3, 0, []), (3, 2, [1]),
                          (3, 2, [0, 1]), (6, 1, [0, 2, 4])])
@pytest.mark.parametrize("compile", [False, True])
def test_apply_zpow_gate(nqubits, target, controls, compile):
    """Check ``apply_zpow`` (including CZPow case)."""
    import itertools
    phase = np.exp(1j * 0.1234)
    qubits = controls[:]
    qubits.append(target)
    qubits.sort()
    matrix = np.ones(2 ** nqubits, dtype=np.complex128)
    for i, conf in enumerate(itertools.product([0, 1], repeat=nqubits)):
        if np.array(conf)[qubits].prod():
            matrix[i] = phase

    state = tensorflow_random_complex((2 ** nqubits,), dtype=tf.float64)

    target_state = np.diag(matrix).dot(state.numpy())

    def apply_operator(state):
      return op.apply_zpow(state, phase, nqubits, target, controls)
    if compile:
        apply_operator = tf.function(apply_operator)
    state = apply_operator(state)

    np.testing.assert_allclose(target_state, state.numpy())


@pytest.mark.parametrize(("nqubits", "targets", "controls",
                          "compile", "einsum_str"),
                         [(3, [0, 1], [], False, "abc,ABab->ABc"),
                          (4, [0, 2], [], True, "abcd,ACac->AbCd"),
                          (3, [0, 1], [2], False, "ab,ABab->AB"),
                          (4, [0, 3], [1], True, "abc,ACac->AbC"),
                          (4, [2, 3], [0], False, "abc,BCbc->aBC"),
                          (5, [4, 1], [2], False, "abcd,BDbd->aBcD"),
                          (6, [1, 3], [0, 4], True, "abcd,ACac->AbCd"),
                          (6, [0, 5], [1, 2, 3], False, "abc,ACac->AbC")])
def test_apply_twoqubit_gate_controlled(nqubits, targets, controls,
                                        compile, einsum_str):
    """Check ``op.apply_twoqubit_gate`` for random gates."""
    state = tensorflow_random_complex((2 ** nqubits,), dtype=tf.float64)
    gate = tensorflow_random_complex((4, 4), dtype=tf.float64)
    gatenp = gate.numpy().reshape(4 * (2,))

    target_state = state.numpy().reshape(nqubits * (2,))
    slicer = nqubits * [slice(None)]
    for c in controls:
        slicer[c] = 1
    slicer = tuple(slicer)
    target_state[slicer] = np.einsum(einsum_str, target_state[slicer], gatenp)
    target_state = target_state.ravel()

    def apply_operator(state):
      return op.apply_twoqubit_gate(state, gate, nqubits, targets, controls)
    if compile:
        apply_operator = tf.function(apply_operator)

    state = apply_operator(state)
    np.testing.assert_allclose(target_state, state.numpy())


@pytest.mark.parametrize(("nqubits", "targets", "controls",
                          "compile", "einsum_str"),
                         [(3, [0, 1], [], False, "abc,ABab->ABc"),
                          (4, [0, 2], [], True, "abcd,ACac->AbCd"),
                          (3, [1, 2], [0], False, "ab,ABab->AB"),
                          (4, [0, 1], [2], False, "abc,ABab->ABc"),
                          (5, [0, 1], [2], False, "abcd,ABab->ABcd"),
                          (5, [3, 4], [2], False, "abcd,CDcd->abCD"),
                          (4, [0, 3], [1], False, "abc,ACac->AbC"),
                          (4, [2, 3], [0], True, "abc,BCbc->aBC"),
                          (5, [1, 4], [2], False, "abcd,BDbd->aBcD"),
                          (6, [1, 3], [0, 4], True, "abcd,ACac->AbCd"),
                          (6, [0, 5], [1, 2, 3], False, "abc,ACac->AbC")])
def test_apply_fsim(nqubits, targets, controls, compile, einsum_str):
    """Check ``op.apply_twoqubit_gate`` for random gates."""
    state = tensorflow_random_complex((2 ** nqubits,), dtype=tf.float64)
    rotation = tensorflow_random_complex((2, 2), dtype=tf.float64)
    phase = tensorflow_random_complex((1,), dtype=tf.float64)

    target_state = state.numpy().reshape(nqubits * (2,))
    gatenp = np.eye(4, dtype=target_state.dtype)
    gatenp[1:3, 1:3] = rotation.numpy()
    gatenp[3, 3] = phase.numpy()[0]
    gatenp = gatenp.reshape(4 * (2,))

    slicer = nqubits * [slice(None)]
    for c in controls:
        slicer[c] = 1
    slicer = tuple(slicer)
    target_state[slicer] = np.einsum(einsum_str, target_state[slicer], gatenp)
    target_state = target_state.ravel()

    gate = tf.concat([tf.reshape(rotation, (4,)), phase], axis=0)
    def apply_operator(state):
      return op.apply_fsim(state, gate, nqubits, targets, controls)
    if compile:
        apply_operator = tf.function(apply_operator)

    state = apply_operator(state)
    np.testing.assert_allclose(target_state, state.numpy())


@pytest.mark.parametrize("compile", [False, True])
def test_apply_swap_with_matrix(compile):
    """Check ``apply_swap`` for two qubits."""
    state = tensorflow_random_complex((2 ** 2,), dtype=tf.float64)
    matrix = np.array([[1, 0, 0, 0],
                       [0, 0, 1, 0],
                       [0, 1, 0, 0],
                       [0, 0, 0, 1]])
    target_state = matrix.dot(state.numpy())

    def apply_operator(state):
      return op.apply_swap(state, 2, targets=[0, 1])
    if compile:
        apply_operator = tf.function(apply_operator)
    state = apply_operator(state)
    np.testing.assert_allclose(target_state, state.numpy())


@pytest.mark.parametrize(("nqubits", "targets", "controls"),
                         [(2, [0, 1], []), (3, [0, 2], []), (4, [1, 3], []),
                          (3, [1, 2], [0]), (4, [0, 2], [1]), (4, [2, 3], [0]),
                          (5, [3, 4], [1, 2]), (6, [1, 4], [0, 2, 5])])
@pytest.mark.parametrize("compile", [False, True])
def test_apply_swap_general(nqubits, targets, controls, compile):
    """Check ``apply_swap`` for more general cases."""
    state = tensorflow_random_complex((2 ** nqubits,), dtype=tf.float64)

    target0, target1 = targets
    for q in controls:
        if q < targets[0]:
            target0 -= 1
        if q < targets[1]:
            target1 -= 1

    target_state = state.numpy().reshape(nqubits * (2,))
    order = list(range(nqubits - len(controls)))
    order[target0], order[target1] = target1, target0
    slicer = tuple(1 if q in controls else slice(None) for q in range(nqubits))
    reduced_state = target_state[slicer]
    reduced_state = np.transpose(reduced_state, order)
    target_state[slicer] = reduced_state

    def apply_operator(state):
      return op.apply_swap(state, nqubits, targets, controls)
    if compile:
        apply_operator = tf.function(apply_operator)
    state = apply_operator(state)
    np.testing.assert_allclose(target_state.ravel(), state.numpy())


# this test fails when compiling due to in-place updates of the state
@pytest.mark.parametrize("gate", ["h", "x", "z", "swap"])
@pytest.mark.parametrize("compile", [False])
def test_custom_op_toy_callback(gate, compile):
    """Check calculating ``callbacks`` using intermediate state values."""
    import functools
    state = tensorflow_random_complex((2 ** 2,), dtype=tf.float64)
    mask = tensorflow_random_complex((2 ** 2,), dtype=tf.float64)

    matrices = {"h": np.array([[1, 1], [1, -1]]) / np.sqrt(2),
                "x": np.array([[0, 1], [1, 0]]),
                "z": np.array([[1, 0], [0, -1]])}
    for k, v in matrices.items():
        matrices[k] = np.kron(v, np.eye(2))
    matrices["swap"] = np.array([[1, 0, 0, 0], [0, 0, 1, 0],
                                 [0, 1, 0, 0], [0, 0, 0, 1]])

    target_state = state.numpy()
    target_c1 = mask.numpy().dot(target_state)
    target_state = matrices[gate].dot(target_state)
    target_c2 = mask.numpy().dot(target_state)
    assert target_c1 != target_c2
    target_callback = [target_c1, target_c2]

    htf = tf.cast(np.array([[1, 1], [1, -1]]) / np.sqrt(2), dtype=state.dtype)
    apply_gate = {"h": functools.partial(op.apply_gate, gate=htf, nqubits=2, target=0),
                  "x": functools.partial(op.apply_x, nqubits=2, target=0),
                  "z": functools.partial(op.apply_z, nqubits=2, target=0),
                  "swap": functools.partial(op.apply_swap, nqubits=2,
                                            targets=[0, 1])}

    def apply_operator(state):
        c1 = tf.reduce_sum(mask * state)
        state0 = apply_gate[gate](state)
        c2 = tf.reduce_sum(mask * state0)
        return state0, tf.stack([c1, c2])
    if compile:
        apply_operator = tf.function(apply_operator)
    state, callback = apply_operator(state)

    np.testing.assert_allclose(target_state, state.numpy())
    np.testing.assert_allclose(target_callback, callback.numpy())