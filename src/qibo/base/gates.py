# -*- coding: utf-8 -*-
# @authors: S. Carrazza and A. Garcia
import numpy as np
from qibo import config
from qibo.config import raise_error
from typing import Dict, List, Optional, Sequence, Tuple

QASM_GATES = {"h": "H", "x": "X", "y": "Y", "z": "Z",
              "rx": "RX", "ry": "RY", "rz": "RZ",
              "u1": "U1", "u2": "U2", "u3": "U3",
              "cx": "CNOT", "swap": "SWAP", "cz": "CZ",
              "crx": "CRX", "cry": "CRY", "crz": "CRZ",
              "cu1": "CU1", "cu3": "CU3",
              "ccx": "TOFFOLI"}
PARAMETRIZED_GATES = {"rx", "ry", "rz", "u1", "u2", "u3",
                      "crx", "cry", "crz", "cu1", "cu3"}


class Gate:
    """The base class for gate implementation.

    All gates should inherit this class.

    Attributes:
        name: Name of the gate.
        target_qubits: Tuple with ids of target qubits.
    """

    def __init__(self):
        self.name = None
        self.is_prepared = False
        self.is_controlled_by = False

        # args for creating gate
        self.init_args = []
        self.init_kwargs = {}

        self._target_qubits = tuple()
        self._control_qubits = set()
        self._nqubits = None
        self._nstates = None
        self._unitary = None
        config.ALLOW_SWITCHERS = False

    @property
    def target_qubits(self) -> Tuple[int]:
        """Tuple with ids of target qubits."""
        return self._target_qubits

    @property
    def control_qubits(self) -> Tuple[int]:
        """Tuple with ids of control qubits sorted in increasing order."""
        return tuple(sorted(self._control_qubits))

    @property
    def qubits(self) -> Tuple[int]:
        """Tuple with ids of all qubits (control and target) that the gate acts."""
        return self.control_qubits + self.target_qubits

    @target_qubits.setter
    def target_qubits(self, qubits: Sequence[int]):
        """Sets control qubits tuple."""
        self._set_target_qubits(qubits)
        self._check_control_target_overlap()

    @control_qubits.setter
    def control_qubits(self, qubits: Sequence[int]):
        """Sets control qubits set."""
        self._set_control_qubits(qubits)
        self._check_control_target_overlap()

    @staticmethod
    def _find_repeated(qubits: Sequence[int]) -> int:
        """Finds the first qubit id that is repeated in a sequence of qubit ids."""
        temp_set = set()
        for qubit in qubits:
            if qubit in temp_set:
                return qubit
            temp_set.add(qubit)

    def _check_control_target_overlap(self):
        """Checks that there are no qubits that are both target and controls."""
        common = set(self._target_qubits) & self._control_qubits
        if common:
            raise_error(ValueError, "{} qubits are both targets and controls for "
                                    "gate {}.".format(common, self.name))

    def _set_target_qubits(self, qubits: Sequence[int]):
        """Helper method for setting target qubits."""
        self._target_qubits = tuple(qubits)
        if len(self._target_qubits) != len(set(qubits)):
            repeated = self._find_repeated(qubits)
            raise_error(ValueError, "Target qubit {} was given twice for gate {}."
                                    "".format(repeated, self.name))

    def _set_control_qubits(self, qubits: Sequence[int]):
        """Helper method for setting control qubits."""
        self._control_qubits = set(qubits)
        if len(self._control_qubits) != len(qubits):
            repeated = self._find_repeated(qubits)
            raise_error(ValueError, "Control qubit {} was given twice for gate {}."
                                    "".format(repeated, self.name))

    def set_targets_and_controls(self, target_qubits: Sequence[int],
                                 control_qubits: Sequence[int]):
        """Sets target and control qubits simultaneously.

        This is used for the reduced qubit updates in the distributed circuits
        because using the individual setters may raise errors due to temporary
        overlap of control and target qubits.
        """
        self._set_target_qubits(target_qubits)
        self._set_control_qubits(control_qubits)
        self._check_control_target_overlap()

    @property
    def nqubits(self) -> int:
        """Number of qubits in the circuit that the gate is part of.

        This is set automatically when the gate is added on a circuit or
        when the gate is called on a state. The user should not set this.
        """
        if self._nqubits is None:
            raise_error(ValueError, "Accessing number of qubits for gate {} but "
                                    "this is not yet set.".format(self))
        return self._nqubits

    @property
    def nstates(self) -> int:
        if self._nstates is None:
            raise_error(ValueError, "Accessing number of qubits for gate {} but "
                                    "this is not yet set.".format(self))
        return self._nstates

    @nqubits.setter
    def nqubits(self, n: int):
        """Sets the total number of qubits that this gate acts on.

        This setter is used by `circuit.add` if the gate is added in a circuit
        or during `__call__` if the gate is called directly on a state.
        The user is not supposed to set `nqubits` by hand.
        """
        if self._nqubits is not None:
            raise_error(RuntimeError, "The number of qubits for this gates is "
                                      "set to {}.".format(self._nqubits))
        self._nqubits = n
        self._nstates = 2**n

    def commutes(self, gate: "Gate") -> bool:
        """Checks if two gates commute.

        Args:
            gate: Gate to check if it commutes with the current gate.

        Returns:
            ``True`` if the gates commute, otherwise ``False``.
        """
        t1 = set(self.target_qubits)
        t2 = set(gate.target_qubits)
        a = self.__class__ == gate.__class__ and t1 == t2
        b = not (t1 & set(gate.qubits) or t2 & set(self.qubits))
        return a or b

    def on_qubits(self, *q) -> "Gate":
        """Creates the same gate targeting different qubits.

        Args:
            q (int): Qubit index (or indeces) that the new gate should act on.
        """
        return self.__class__(*q, **self.init_kwargs)

    def decompose(self) -> List["Gate"]:
        """Decomposes multi-control gates to gates supported by OpenQASM.

        Decompositions are based on `arXiv:9503016 <https://arxiv.org/abs/quant-ph/9503016>`_.

        Args:
            free: Ids of free qubits to use for the gate decomposition.

        Returns:
            List with gates that have the same effect as applying the original gate.
        """
        # FIXME: Implement this method for all gates not supported by OpenQASM.
        # If the method is not implemented this returns a deep copy of the
        # original gate
        return [self.__class__(*self.init_args, **self.init_kwargs)]

    def _dagger(self) -> "Gate":
        """Helper method for :meth:`qibo.base.gates.Gate.dagger`."""
        return self.__class__(*self.init_args, **self.init_kwargs)

    def dagger(self) -> "Gate":
        """Returns the dagger (conjugate transpose) of the gate.

        Returns:
            A :class:`qibo.base.gates.Gate` object representing the dagger of
            the original gate.
        """
        new_gate = self._dagger()
        new_gate.is_controlled_by = self.is_controlled_by
        new_gate.control_qubits = self.control_qubits
        return new_gate

    def controlled_by(self, *qubits: int) -> "Gate":
        """Controls the gate on (arbitrarily many) qubits.

        Args:
            *qubits (int): Ids of the qubits that the gate will be controlled on.

        Returns:
            A :class:`qibo.base.gates.Gate` object in with the corresponding
            gate being controlled in the given qubits.
        """
        if self.control_qubits:
            raise_error(RuntimeError, "Cannot use `controlled_by` method on gate {} "
                                      "because it is already controlled by {}."
                                      "".format(self, self.control_qubits))
        if self.is_prepared:
            raise_error(RuntimeError, "Cannot use controlled_by on a prepared gate.")
        if qubits:
            self.is_controlled_by = True
            self.control_qubits = qubits
        return self

    @property
    def unitary(self):
        """Unitary matrix corresponding to the gate's action on a state vector.

        This matrix is not necessarily used by ``__call__`` when applying the
        gate to a state vector.
        """
        if len(self.qubits) > 2:
            raise_error(NotImplementedError, "Cannot calculate unitary matrix for "
                                             "gates that target more than two qubits.")
        if self._unitary is None:
            self._unitary = self.construct_unitary() # pylint: disable=E1101
        return self._unitary

    def __matmul__(self, other: "Gate") -> "Gate":
        """Gate multiplication."""
        if self.qubits != other.qubits:
            raise_error(NotImplementedError,
                        "Cannot multiply gates that target different qubits.")
        if self.__class__.__name__ == other.__class__.__name__:
            square_identity = {"H", "X", "Y", "Z", "CNOT", "CZ", "SWAP"}
            if self.__class__.__name__ in square_identity:
                return self.module.I(*self.qubits)
        return self.module.Unitary(self.unitary @ other.unitary, *self.qubits)

    def __rmatmul__(self, other: "TensorflowGate") -> "TensorflowGate":
        return self.__matmul__(other)


class SpecialGate(Gate):
    """Abstract class for special gates.

    Special gates are currently the :class:`qibo.base.gates.CallbackGate` and
    :class:`qibo.base.gates.Flatten`.
    """

    def commutes(self, gate):
        return False

    def on_qubits(self, *q):
        raise_error(NotImplementedError,
                    "Cannot use special gates on subroutines.")


class ParametrizedGate(Gate):
    """Base class for parametrized gates.

    Implements the basic functionality of parameter setters and getters.
    """

    def __init__(self):
        super(ParametrizedGate, self).__init__()
        self._theta = None
        self.nparams = 1

    @property
    def parameter(self):
        return self._theta

    @parameter.setter
    def parameter(self, x):
        self._unitary = None
        self._theta = x


class H(Gate):
    """The Hadamard gate.

    Args:
        q (int): the qubit id number.
    """

    def __init__(self, q):
        super().__init__()
        self.name = "h"
        self.target_qubits = (q,)
        self.init_args = [q]


class X(Gate):
    """The Pauli X gate.

    Args:
        q (int): the qubit id number.
    """

    def __init__(self, q):
        super().__init__()
        self.name = "x"
        self.target_qubits = (q,)
        self.init_args = [q]

    def controlled_by(self, *q):
        """Fall back to CNOT and Toffoli if there is one or two controls."""
        if len(q) == 1:
            gate = self.module.CNOT(q[0], self.target_qubits[0])
        elif len(q) == 2:
            gate = self.module.TOFFOLI(q[0], q[1], self.target_qubits[0])
        else:
            gate = super(X, self).controlled_by(*q)
        return gate

    def decompose(self, *free: int, use_toffolis: bool = True) -> List[Gate]: # pylint: disable=W0221
        """Decomposes multi-control ``X`` gate to one-qubit, ``CNOT`` and ``TOFFOLI`` gates.

        Args:
            free: Ids of free qubits to use for the gate decomposition.
            use_toffolis: If ``True`` the decomposition contains only ``TOFFOLI`` gates.
                If ``False`` a congruent representation is used for ``TOFFOLI`` gates.
                See :class:`qibo.base.gates.TOFFOLI` for more details on this representation.

        Returns:
            List with one-qubit, ``CNOT`` and ``TOFFOLI`` gates that have the
            same effect as applying the original multi-control gate.
        """
        if set(free) & set(self.qubits):
            raise_error(ValueError, "Cannot decompose multi-control X gate if free "
                                    "qubits coincide with target or controls.")
        if self._nqubits is not None:
            for q in free:
                if q >= self.nqubits:
                    raise_error(ValueError, "Gate acts on {} qubits but {} was given "
                                            "as free qubit.".format(self.nqubits, q))

        controls = self.control_qubits
        target = self.target_qubits[0]
        m = len(controls)
        if m < 3:
            return [self.__class__(target).controlled_by(*controls)]

        decomp_gates = []
        n = m + 1 + len(free)
        toffoli_cls = self.module.TOFFOLI
        if (n >= 2 * m - 1) and (m >= 3):
            gates1 = [toffoli_cls(controls[m - 2 - i],
                                  free[m - 4 - i],
                                  free[m - 3 - i]
                                  ).congruent(use_toffolis=use_toffolis)
                      for i in range(m - 3)]
            gates2 = toffoli_cls(controls[0], controls[1], free[0]
                                 ).congruent(use_toffolis=use_toffolis)
            first_toffoli = toffoli_cls(controls[m - 1], free[m - 3], target)

            decomp_gates.append(first_toffoli)
            for gates in gates1:
                decomp_gates.extend(gates)
            decomp_gates.extend(gates2)
            for gates in gates1[::-1]:
                decomp_gates.extend(gates)

        elif len(free) >= 1:
            m1 = n // 2
            free1 = controls[m1:] + (target,) + tuple(free[1:])
            x1 = self.__class__(free[0]).controlled_by(*controls[:m1])
            part1 = x1.decompose(*free1, use_toffolis=use_toffolis)

            free2 = controls[:m1] + tuple(free[1:])
            controls2 = controls[m1:] + (free[0],)
            x2 = self.__class__(target).controlled_by(*controls2)
            part2 = x2.decompose(*free2, use_toffolis=use_toffolis)

            decomp_gates = [*part1, *part2]

        else: # pragma: no cover
            # impractical case
            raise_error(NotImplementedError, "X decomposition not implemented "
                                             "for zero free qubits.")

        decomp_gates.extend(decomp_gates)
        return decomp_gates


class Y(Gate):
    """The Pauli Y gate.

    Args:
        q (int): the qubit id number.
    """

    def __init__(self, q):
        super(Y, self).__init__()
        self.name = "y"
        self.target_qubits = (q,)
        self.init_args = [q]


class Z(Gate):
    """The Pauli Z gate.

    Args:
        q (int): the qubit id number.
    """

    def __init__(self, q):
        super(Z, self).__init__()
        self.name = "z"
        self.target_qubits = (q,)
        self.init_args = [q]

    def controlled_by(self, *q):
        """Fall back to CZ if there is only one control."""
        if len(q) == 1:
            gate = getattr(self.module, "CZ")(q[0], self.target_qubits[0])
        else:
            gate = super(Z, self).controlled_by(*q)
        return gate


class I(Gate):
    """The identity gate.

    Args:
        *q (int): the qubit id numbers.
    """

    def __init__(self, *q):
        super(I, self).__init__()
        self.name = "identity"
        self.target_qubits = tuple(q)
        self.init_args = q


class Collapse(Gate):
    """Gate that collapses the state vector according to a measurement.

    Args:
        *q (int): the qubit id numbers that were measured.
        result (int/list): measured result for each qubit. Should be 0 or 1.
            If a ``list`` is given, it should have the same length as the
            number of qubits measured. If an ``int`` is given, the same
            result is used for all qubits measured.
    """

    def __init__(self, *q, result=0):
        super(Collapse, self).__init__()
        if isinstance(result, int):
            result = len(q) * [result]
        self.name = "collapse"
        self.target_qubits = tuple(q)
        self._result = None

        self.init_args = q
        self.init_kwargs = {"result": result}
        self.sorted_qubits = sorted(q)
        self.result = result
        # Flag that is turned ``False`` automatically if this gate is used in a
        # ``TensorflowDistributedCircuit`` in order to skip the normalization.
        self.normalize = True

    @property
    def result(self):
        """Returns the result list in proper order after sorting the qubits."""
        return self._result

    @result.setter
    def result(self, res):
        res = self._result_to_list(res) # pylint: disable=E1111
        if len(self.target_qubits) != len(res):
            raise_error(ValueError, "Collapse gate was created on {} qubits "
                                    "but {} result values were given."
                                    "".format(len(self.target_qubits), len(res)))
        resdict = {}
        for q, r in zip(self.target_qubits, res):
            if r not in {0, 1}:
                raise_error(ValueError, "Result values should be 0 or 1 but "
                                        "{} was given.".format(r))
            resdict[q] = r

        self._result = [resdict[q] for q in self.sorted_qubits]
        self.init_kwargs = {"result": res}

    def controlled_by(self, *q): # pragma: no cover # pylint: disable=W0613
        """"""
        raise_error(NotImplementedError, "Collapse gates cannot be controlled.")


class M(Gate):
    """The Measure Z gate.

    Args:
        *q (int): id numbers of the qubits to measure.
            It is possible to measure multiple qubits using ``gates.M(0, 1, 2, ...)``.
            If the qubits to measure are held in an iterable (eg. list) the ``*``
            operator can be used, for example ``gates.M(*[0, 1, 4])`` or
            ``gates.M(*range(5))``.
        register_name (str): Optional name of the register to distinguish it
            from other registers when used in circuits.
        p0 (dict): Optional bitflip probability map. Can be:
            A dictionary that maps each measured qubit to the probability
            that it is flipped, a list or tuple that has the same length
            as the tuple of measured qubits or a single float number.
            If a single float is given the same probability will be used
            for all qubits.
        p1 (dict): Optional bitflip probability map for asymmetric bitflips.
            Same as ``p0`` but controls the 1->0 bitflip probability.
            If ``p1`` is ``None`` then ``p0`` will be used both for 0->1 and
            1->0 bitflips.
    """
    from qibo.base import measurements

    def __init__(self, *q, register_name: Optional[str] = None,
                 p0: Optional["ProbsType"] = None,
                 p1: Optional["ProbsType"] = None):
        super(M, self).__init__()
        self.name = "measure"
        self.target_qubits = q
        self.register_name = register_name

        self.init_args = q
        self.init_kwargs = {"register_name": register_name, "p0": p0, "p1": p1}

        self._unmeasured_qubits = None # Tuple
        self._reduced_target_qubits = None # List
        if p1 is None: p1 = p0
        if p0 is None: p0 = p1
        self.bitflip_map = (self._get_bitflip_map(p0),
                            self._get_bitflip_map(p1))

    def _get_bitflip_map(self, p: Optional["ProbsType"] = None
                         ) -> Dict[int, float]:
        """Creates dictionary with bitflip probabilities."""
        if p is None:
            return {q: 0 for q in self.qubits}
        pt = self.measurements.GateResult._get_bitflip_tuple(self.qubits, p)
        return {q: p for q, p in zip(self.qubits, pt)}

    def _add(self, gate: "M"):
        """Adds target qubits to a measurement gate.

        This method is only used for creating the global measurement gate used
        by the `models.Circuit`.
        The user is not supposed to use this method and a `ValueError` is
        raised if he does so.

        Args:
            gate: Measurement gate to add its qubits in the current gate.
        """
        if self._unmeasured_qubits is not None:
            raise_error(RuntimeError, "Cannot add qubits to a measurement "
                                      "gate that was executed.")
        assert isinstance(gate, self.__class__)
        self.target_qubits += gate.target_qubits
        self.bitflip_map[0].update(gate.bitflip_map[0])
        self.bitflip_map[1].update(gate.bitflip_map[1])

    def _set_unmeasured_qubits(self):
        if self._nqubits is None:
            raise_error(RuntimeError, "Cannot calculate set of unmeasured "
                                      "qubits if the number of qubits in the "
                                      "circuit is unknown.")
        if self._unmeasured_qubits is not None:
            raise_error(RuntimeError, "Cannot recalculate unmeasured qubits.")
        target_qubits = set(self.target_qubits)
        unmeasured_qubits = []
        reduced_target_qubits = dict()
        for i in range(self.nqubits):
            if i in target_qubits:
                reduced_target_qubits[i] = i - len(unmeasured_qubits)
            else:
                unmeasured_qubits.append(i)

        self._unmeasured_qubits = tuple(unmeasured_qubits)
        self._reduced_target_qubits = list(reduced_target_qubits[i]
                                           for i in self.target_qubits)

    @property
    def unmeasured_qubits(self) -> Tuple[int]:
        """Tuple with ids of unmeasured qubits sorted in increasing order.

        This is useful when tracing out unmeasured qubits to calculate
        probabilities.
        """
        if self._unmeasured_qubits is None:
            self._set_unmeasured_qubits()
        return self._unmeasured_qubits

    @property
    def reduced_target_qubits(self) -> List[int]:
        if self._unmeasured_qubits is None:
            self._set_unmeasured_qubits()
        return self._reduced_target_qubits

    def controlled_by(self, *q): # pylint: disable=W0613
        """"""
        raise_error(NotImplementedError, "Measurement gates cannot be controlled.")


class _Rn_(ParametrizedGate):
    """Abstract class for defining the RX, RY and RZ rotations.

    Args:
        q (int): the qubit id number.
        theta (float): the rotation angle.
    """
    axis = "n"

    def __init__(self, q, theta):
        super(_Rn_, self).__init__()
        self.name = "r{}".format(self.axis)
        self.target_qubits = (q,)
        self.parameter = theta

        self.init_args = [q]
        self.init_kwargs = {"theta": theta}

    def _dagger(self) -> "Gate":
        """"""
        return self.__class__(self.target_qubits[0], -self.parameter)

    def controlled_by(self, *q):
        """Fall back to CRn if there is only one control."""
        if len(q) == 1:
            gate = getattr(self.module, "CR{}".format(self.axis.capitalize()))(
              q[0], self.target_qubits[0], **self.init_kwargs)
        else:
            gate = super(_Rn_, self).controlled_by(*q)
        return gate


class RX(_Rn_):
    """Rotation around the X-axis of the Bloch sphere.

    Corresponds to the following unitary matrix

    .. math::
        \\begin{pmatrix}
        \\cos \\frac{\\theta }{2}  &
        -i\\sin \\frac{\\theta }{2} \\\\
        -i\\sin \\frac{\\theta }{2}  &
        \\cos \\frac{\\theta }{2} \\\\
        \\end{pmatrix}

    Args:
        q (int): the qubit id number.
        theta (float): the rotation angle.
    """
    axis = "x"


class RY(_Rn_):
    """Rotation around the Y-axis of the Bloch sphere.

    Corresponds to the following unitary matrix

    .. math::
        \\begin{pmatrix}
        \\cos \\frac{\\theta }{2}  &
        -\\sin \\frac{\\theta }{2} \\\\
        \\sin \\frac{\\theta }{2}  &
        \\cos \\frac{\\theta }{2} \\\\
        \\end{pmatrix}

    Args:
        q (int): the qubit id number.
        theta (float): the rotation angle.
    """
    axis = "y"


class RZ(_Rn_):
    """Rotation around the Z-axis of the Bloch sphere.

    Corresponds to the following unitary matrix

    .. math::
        \\begin{pmatrix}
        e^{-i \\theta / 2} & 0 \\\\
        0 & e^{i \\theta / 2} \\\\
        \\end{pmatrix}

    Args:
        q (int): the qubit id number.
        theta (float): the rotation angle.
    """
    axis = "z"


class _Un_(ParametrizedGate):
    """Abstract class for defining the U1, U2 and U3 gates.

    Args:
        q (int): the qubit id number.
    """
    order = 0

    def __init__(self, q):
        super(_Un_, self).__init__()
        self.name = "u{}".format(self.order)
        self.nparams = self.order
        self.target_qubits = (q,)
        self.init_args = [q]

    def controlled_by(self, *q):
        """Fall back to CUn if there is only one control."""
        if len(q) == 1:
            gate = getattr(self.module, "CU{}".format(self.order))(
              q[0], self.target_qubits[0], **self.init_kwargs)
        else:
            gate = super(_Un_, self).controlled_by(*q)
        return gate


class U1(_Un_):
    """First general unitary gate.

    Corresponds to the following unitary matrix

    .. math::
        \\begin{pmatrix}
        1 & 0 \\\\
        0 & e^{i \\theta} \\\\
        \\end{pmatrix}

    Args:
        q (int): the qubit id number.
        theta (float): the rotation angle.
    """
    order = 1

    def __init__(self, q, theta):
        super(U1, self).__init__(q)
        self.parameter = theta
        self.init_kwargs = {"theta": theta}

    def _dagger(self) -> "Gate":
        """"""
        return self.__class__(self.target_qubits[0], -self.parameter)


class U2(_Un_):
    """Second general unitary gate.

    Corresponds to the following unitary matrix

    .. math::
        \\frac{1}{\\sqrt{2}}
        \\begin{pmatrix}
        e^{-i(\\phi + \\lambda )/2} & -e^{-i(\\phi - \\lambda )/2} \\\\
        e^{i(\\phi - \\lambda )/2} & e^{i (\\phi + \\lambda )/2} \\\\
        \\end{pmatrix}

    Args:
        q (int): the qubit id number.
        phi (float): first rotation angle.
        lamb (float): second rotation angle.
    """
    order = 2

    def __init__(self, q, phi, lam):
        super(U2, self).__init__(q)
        self._phi, self._lam = None, None
        self.init_kwargs = {"phi": phi, "lam": lam}
        self.parameter = phi, lam

    def _dagger(self) -> "Gate":
        """"""
        phi = np.pi - self._lam
        lam = - np.pi - self._phi
        return self.__class__(self.target_qubits[0], phi, lam)

    @property
    def parameter(self):
        return self._phi, self._lam

    @parameter.setter
    def parameter(self, x):
        self._unitary = None
        self._phi, self._lam = x


class U3(_Un_):
    """Third general unitary gate.

    Corresponds to the following unitary matrix

    .. math::
        \\begin{pmatrix}
        e^{-i(\\phi + \\lambda )/2}\\cos\\left (\\frac{\\theta }{2}\\right ) & -e^{-i(\\phi - \\lambda )/2}\\sin\\left (\\frac{\\theta }{2}\\right ) \\\\
        e^{i(\\phi - \\lambda )/2}\\sin\\left (\\frac{\\theta }{2}\\right ) & e^{i (\\phi + \\lambda )/2}\\cos\\left (\\frac{\\theta }{2}\\right ) \\\\
        \\end{pmatrix}

    Args:
        q (int): the qubit id number.
        theta (float): first rotation angle.
        phi (float): second rotation angle.
        lamb (float): third rotation angle.
    """
    order = 3

    def __init__(self, q, theta, phi, lam):
        super(U3, self).__init__(q)
        self._theta, self._phi, self._lam = None, None, None
        self.init_kwargs = {"theta": theta, "phi": phi, "lam": lam}
        self.parameter = theta, phi, lam

    def _dagger(self) -> "Gate":
        """"""
        theta, lam, phi = tuple(-x for x in self.parameter)
        return self.__class__(self.target_qubits[0], theta, phi, lam)

    @property
    def parameter(self):
        return self._theta, self._phi, self._lam

    @parameter.setter
    def parameter(self, x):
        self._unitary = None
        self._theta, self._phi, self._lam = x


class ZPow(Gate): # pragma: no cover
    """Equivalent to :class:`qibo.base.gates.U1`.

    Implemented to maintain compatibility with previous versions.
    Corresponds to the following unitary matrix

    .. math::
        \\begin{pmatrix}
        1 & 0 \\\\
        0 & e^{i \\theta} \\\\
        \\end{pmatrix}

    Args:
        q (int): the qubit id number.
        theta (float): the rotation angle.
    """
    def __new__(cls, q, theta): # pragma: no cover
        # code is not tested as it is substituted in `tensorflow` gates
        return U1(q, theta)


class CNOT(Gate):
    """The Controlled-NOT gate.

    Corresponds to the following unitary matrix

    .. math::
        \\begin{pmatrix}
        1 & 0 & 0 & 0 \\\\
        0 & 1 & 0 & 0 \\\\
        0 & 0 & 0 & 1 \\\\
        0 & 0 & 1 & 0 \\\\
        \\end{pmatrix}

    Args:
        q0 (int): the control qubit id number.
        q1 (int): the target qubit id number.
    """

    def __init__(self, q0, q1):
        super(CNOT, self).__init__()
        self.name = "cx"
        self.control_qubits = (q0,)
        self.target_qubits = (q1,)
        self.init_args = [q0, q1]

    def decompose(self) -> List[Gate]:
        q0, q1 = self.control_qubits[0], self.target_qubits[0]
        return [self.__class__(q0, q1)]


class CZ(Gate):
    """The Controlled-Phase gate.

    Corresponds to the following unitary matrix

    .. math::
        \\begin{pmatrix}
        1 & 0 & 0 & 0 \\\\
        0 & 1 & 0 & 0 \\\\
        0 & 0 & 1 & 0 \\\\
        0 & 0 & 0 & -1 \\\\
        \\end{pmatrix}

    Args:
        q0 (int): the control qubit id number.
        q1 (int): the target qubit id number.
    """

    def __init__(self, q0, q1):
        super(CZ, self).__init__()
        self.name = "cz"
        self.control_qubits = (q0,)
        self.target_qubits = (q1,)
        self.init_args = [q0, q1]


class _CRn_(ParametrizedGate):
    """Abstract method for defining the CU1, CU2 and CU3 gates.

    Args:
        q0 (int): the control qubit id number.
        q1 (int): the target qubit id number.
        theta (float): the rotation angle.
    """
    axis = "n"

    def __init__(self, q0, q1, theta):
        super(_CRn_, self).__init__()
        self.name = "cr{}".format(self.axis)
        self.control_qubits = (q0,)
        self.target_qubits = (q1,)
        self.parameter = theta

        self.init_args = [q0, q1]
        self.init_kwargs = {"theta": theta}

    def _dagger(self) -> "Gate":
        """"""
        q0 = self.control_qubits[0]
        q1 = self.target_qubits[0]
        return self.__class__(q0, q1, -self.parameter)


class CRX(_CRn_):
    """Controlled rotation around the X-axis for the Bloch sphere.

    Corresponds to the following unitary matrix

    .. math::
        \\begin{pmatrix}
        1 & 0 & 0 & 0 \\\\
        0 & 1 & 0 & 0 \\\\
        0 & 0 & \\cos \\frac{\\theta }{2}  & -i\\sin \\frac{\\theta }{2} \\\\
        0 & 0 & -i\\sin \\frac{\\theta }{2}  & \\cos \\frac{\\theta }{2} \\\\
        \\end{pmatrix}

    Args:
        q0 (int): the control qubit id number.
        q1 (int): the target qubit id number.
        theta (float): the rotation angle.
    """
    axis = "x"


class CRY(_CRn_):
    """Controlled rotation around the X-axis for the Bloch sphere.

    Corresponds to the following unitary matrix

    .. math::
        \\begin{pmatrix}
        1 & 0 & 0 & 0 \\\\
        0 & 1 & 0 & 0 \\\\
        0 & 0 & \\cos \\frac{\\theta }{2}  & -\\sin \\frac{\\theta }{2} \\\\
        0 & 0 & \\sin \\frac{\\theta }{2}  & \\cos \\frac{\\theta }{2} \\\\
        \\end{pmatrix}

    Note that this differs from the :class:`qibo.base.gates.RZ` gate.

    Args:
        q0 (int): the control qubit id number.
        q1 (int): the target qubit id number.
        theta (float): the rotation angle.
    """
    axis = "y"


class CRZ(_CRn_):
    """Controlled rotation around the X-axis for the Bloch sphere.

    Corresponds to the following unitary matrix

    .. math::
        \\begin{pmatrix}
        1 & 0 & 0 & 0 \\\\
        0 & 1 & 0 & 0 \\\\
        0 & 0 & e^{-i \\theta / 2} & 0 \\\\
        0 & 0 & 0 & e^{i \\theta / 2} \\\\
        \\end{pmatrix}

    Args:
        q0 (int): the control qubit id number.
        q1 (int): the target qubit id number.
        theta (float): the rotation angle.
    """
    axis = "z"


class _CUn_(ParametrizedGate):
    """Abstract method for defining the CU1, CU2 and CU3 gates.

    Args:
        q0 (int): the control qubit id number.
        q1 (int): the target qubit id number.
    """
    order = 0

    def __init__(self, q0, q1):
        super(_CUn_, self).__init__()
        self.name = "cu{}".format(self.order)
        self.nparams = self.order
        self.control_qubits = (q0,)
        self.target_qubits = (q1,)
        self.init_args = [q0, q1]


class CU1(_CUn_):
    """Controlled first general unitary gate.

    Corresponds to the following unitary matrix

    .. math::
        \\begin{pmatrix}
        1 & 0 & 0 & 0 \\\\
        0 & 1 & 0 & 0 \\\\
        0 & 0 & 1 & 0 \\\\
        0 & 0 & 0 & e^{i \\theta } \\\\
        \\end{pmatrix}

    Note that this differs from the :class:`qibo.base.gates.CRZ` gate.

    Args:
        q0 (int): the control qubit id number.
        q1 (int): the target qubit id number.
        theta (float): the rotation angle.
    """
    order = 1

    def __init__(self, q0, q1, theta):
        super(CU1, self).__init__(q0, q1)
        self.parameter = theta
        self.init_kwargs = {"theta": theta}

    def _dagger(self) -> "Gate":
        """"""
        q0 = self.control_qubits[0]
        q1 = self.target_qubits[0]
        return self.__class__(q0, q1, -self.parameter)


class CU2(_CUn_):
    """Controlled second general unitary gate.

    Corresponds to the following unitary matrix

    .. math::
        \\frac{1}{\\sqrt{2}}
        \\begin{pmatrix}
        1 & 0 & 0 & 0 \\\\
        0 & 1 & 0 & 0 \\\\
        0 & 0 & e^{-i(\\phi + \\lambda )/2} & -e^{-i(\\phi - \\lambda )/2} \\\\
        0 & 0 & e^{i(\\phi - \\lambda )/2} & e^{i (\\phi + \\lambda )/2} \\\\
        \\end{pmatrix}

    Args:
        q0 (int): the control qubit id number.
        q1 (int): the target qubit id number.
        phi (float): first rotation angle.
        lamb (float): second rotation angle.
    """
    order = 2

    def __init__(self, q0, q1, phi, lam):
        super(CU2, self).__init__(q0, q1)
        self._phi, self._lam = None, None
        self.init_kwargs = {"phi": phi, "lam": lam}
        self.parameter = phi, lam

    def _dagger(self) -> "Gate":
        """"""
        q0 = self.control_qubits[0]
        q1 = self.target_qubits[0]
        phi = np.pi - self._lam
        lam = - np.pi - self._phi
        return self.__class__(q0, q1, phi, lam)

    @property
    def parameter(self):
        return self._phi, self._lam

    @parameter.setter
    def parameter(self, x):
        self._unitary = None
        self._phi, self._lam = x


class CU3(_CUn_):
    """Controlled third general unitary gate.

    Corresponds to the following unitary matrix

    .. math::
        \\begin{pmatrix}
        1 & 0 & 0 & 0 \\\\
        0 & 1 & 0 & 0 \\\\
        0 & 0 & e^{-i(\\phi + \\lambda )/2}\\cos\\left (\\frac{\\theta }{2}\\right ) & -e^{-i(\\phi - \\lambda )/2}\\sin\\left (\\frac{\\theta }{2}\\right ) \\\\
        0 & 0 & e^{i(\\phi - \\lambda )/2}\\sin\\left (\\frac{\\theta }{2}\\right ) & e^{i (\\phi + \\lambda )/2}\\cos\\left (\\frac{\\theta }{2}\\right ) \\\\
        \\end{pmatrix}

    Args:
        q0 (int): the control qubit id number.
        q1 (int): the target qubit id number.
        theta (float): first rotation angle.
        phi (float): second rotation angle.
        lamb (float): third rotation angle.
    """
    order = 3

    def __init__(self, q0, q1, theta, phi, lam):
        super(CU3, self).__init__(q0, q1)
        self._theta, self._phi, self._lam = None, None, None
        self.init_kwargs = {"theta": theta, "phi": phi, "lam": lam}
        self.parameter = theta, phi, lam

    def _dagger(self) -> "Gate":
        """"""
        q0 = self.control_qubits[0]
        q1 = self.target_qubits[0]
        theta, lam, phi = tuple(-x for x in self.parameter)
        return self.__class__(q0, q1, theta, phi, lam)

    @property
    def parameter(self):
        return self._theta, self._phi, self._lam

    @parameter.setter
    def parameter(self, x):
        self._unitary = None
        self._theta, self._phi, self._lam = x


class CZPow(Gate): # pragma: no cover
    """Equivalent to :class:`qibo.base.gates.CU1`.

    Implemented to maintain compatibility with previous versions.
    Corresponds to the following unitary matrix

    .. math::
        \\begin{pmatrix}
        1 & 0 & 0 & 0 \\\\
        0 & 1 & 0 & 0 \\\\
        0 & 0 & 1 & 0 \\\\
        0 & 0 & 0 & e^{i \\theta } \\\\
        \\end{pmatrix}

    Args:
        q0 (int): the control qubit id number.
        q1 (int): the target qubit id number.
        theta (float): the rotation angle.
    """
    def __new__(cls, q0, q1, theta): # pragma: no cover
        # code is not tested as it is substituted in `tensorflow` gates
        return CU1(q0, q1, theta)


class SWAP(Gate):
    """The swap gate.

    Corresponds to the following unitary matrix

    .. math::
        \\begin{pmatrix}
        1 & 0 & 0 & 0 \\\\
        0 & 0 & 1 & 0 \\\\
        0 & 1 & 0 & 0 \\\\
        0 & 0 & 0 & 1 \\\\
        \\end{pmatrix}

    Args:
        q0 (int): the first qubit to be swapped id number.
        q1 (int): the second qubit to be swapped id number.
    """

    def __init__(self, q0, q1):
        super(SWAP, self).__init__()
        self.name = "swap"
        self.target_qubits = (q0, q1)
        self.init_args = [q0, q1]


class fSim(ParametrizedGate):
    """The fSim gate defined in `arXiv:2001.08343 <https://arxiv.org/abs/2001.08343>`_.

    Corresponds to the following unitary matrix

    .. math::
        \\begin{pmatrix}
        1 & 0 & 0 & 0 \\\\
        0 & \\cos \\theta & -i\\sin \\theta & 0 \\\\
        0 & -i\\sin \\theta & \\cos \\theta & 0 \\\\
        0 & 0 & 0 & e^{-i \\phi } \\\\
        \\end{pmatrix}

    Args:
        q0 (int): the first qubit to be swapped id number.
        q1 (int): the second qubit to be swapped id number.
        theta (float): Angle for the one-qubit rotation.
        phi (float): Angle for the |11> phase.
    """
    # TODO: Check how this works with QASM.

    def __init__(self, q0, q1, theta, phi):
        super(fSim, self).__init__()
        self.name = "fsim"
        self.target_qubits = (q0, q1)
        self._phi = None
        self.nparams = 2
        self.parameter = theta, phi

        self.init_args = [q0, q1]
        self.init_kwargs = {"theta": theta, "phi": phi}

    def _dagger(self) -> "Gate":
        """"""
        q0, q1 = self.target_qubits
        return self.__class__(q0, q1, *(-x for x in self.parameter))

    @property
    def parameter(self):
        return self._theta, self._phi

    @parameter.setter
    def parameter(self, x):
        self._unitary = None
        self._theta, self._phi = x


class GeneralizedfSim(ParametrizedGate):
    """The fSim gate with a general rotation.

    Corresponds to the following unitary matrix

    .. math::
        \\begin{pmatrix}
        1 & 0 & 0 & 0 \\\\
        0 & R_{00} & R_{01} & 0 \\\\
        0 & R_{10} & R_{11} & 0 \\\\
        0 & 0 & 0 & e^{-i \\phi } \\\\
        \\end{pmatrix}

    Args:
        q0 (int): the first qubit to be swapped id number.
        q1 (int): the second qubit to be swapped id number.
        unitary (np.ndarray): Unitary that corresponds to the one-qubit rotation.
        phi (float): Angle for the |11> phase.
    """

    def __init__(self, q0, q1, unitary, phi):
        super(GeneralizedfSim, self).__init__()
        self.name = "generalizedfsim"
        self.target_qubits = (q0, q1)

        self._phi = None
        self.__unitary = None
        self.nparams = 2
        self.parameter = unitary, phi

        self.init_args = [q0, q1]
        self.init_kwargs = {"unitary": unitary, "phi": phi}

    @property
    def parameter(self):
        return self.__unitary, self._phi

    @parameter.setter
    def parameter(self, x):
        shape = tuple(x[0].shape)
        if shape != (2, 2):
            raise_error(ValueError, "Invalid shape {} of rotation for generalized "
                                    "fSim gate".format(shape))
        self._unitary = None
        self.__unitary, self._phi = x


class TOFFOLI(Gate):
    """The Toffoli gate.

    Args:
        q0 (int): the first control qubit id number.
        q1 (int): the second control qubit id number.
        q2 (int): the target qubit id number.
    """

    def __init__(self, q0, q1, q2):
        super(TOFFOLI, self).__init__()
        self.name = "ccx"
        self.control_qubits = (q0, q1)
        self.target_qubits = (q2,)
        self.init_args = [q0, q1, q2]

    def decompose(self) -> List[Gate]:
        c0, c1 = self.control_qubits
        t = self.target_qubits[0]
        return [self.__class__(c0, c1, t)]

    def congruent(self, use_toffolis: bool = True) -> List[Gate]:
        """Congruent representation of ``TOFFOLI`` gate.

        This is a helper method for the decomposition of multi-control ``X`` gates.
        The congruent representation is based on Sec. 6.2 of
        `arXiv:9503016 <https://arxiv.org/abs/quant-ph/9503016>`_.
        The sequence of the gates produced here has the same effect as ``TOFFOLI``
        with the phase of the |101> state reversed.

        Args:
            use_toffolis: If ``True`` a single ``TOFFOLI`` gate is returned.
                If ``False`` the congruent representation is returned.

        Returns:
            List with ``RY`` and ``CNOT`` gates that have the same effect as
            applying the original ``TOFFOLI`` gate.
        """
        if use_toffolis:
            return self.decompose()
        control0, control1 = self.control_qubits
        target = self.target_qubits[0]
        ry = self.module.RY
        cnot = self.module.CNOT
        return [ry(target, -np.pi / 4), cnot(control1, target),
                ry(target, -np.pi / 4), cnot(control0, target),
                ry(target, np.pi / 4), cnot(control1, target),
                ry(target, np.pi / 4)]


class Unitary(ParametrizedGate):
    """Arbitrary unitary gate.

    Args:
        unitary: Unitary matrix as a tensor supported by the backend.
            Note that there is no check that the matrix passed is actually
            unitary. This allows the user to create non-unitary gates.
        *q (int): Qubit id numbers that the gate acts on.
        name (str): Optional name for the gate.
    """

    def __init__(self, unitary, *q, name: Optional[str] = None):
        super(Unitary, self).__init__()
        self.name = "Unitary" if name is None else name
        self.target_qubits = tuple(q)

        self.__unitary = None
        self.parameter = unitary
        self.nparams = int(tuple(unitary.shape)[0]) ** 2

        self.init_args = [unitary] + list(q)
        self.init_kwargs = {"name": name}

    @property
    def rank(self) -> int:
        return len(self.target_qubits)

    def on_qubits(self, *q) -> "Gate":
        args = [self.init_args[0]]
        args.extend(q)
        return self.__class__(*args, **self.init_kwargs)

    @property
    def parameter(self):
        return self.__unitary

    @parameter.setter
    def parameter(self, x):
        shape = tuple(x.shape)
        true_shape = (2 ** self.rank, 2 ** self.rank)
        if shape == true_shape:
            self.__unitary = x
        elif shape == (2 ** (2 * self.rank),):
            self.__unitary = x.reshape(true_shape)
        else:
            raise_error(ValueError, "Invalid shape {} of unitary matrix acting on "
                                    "{} target qubits.".format(shape, self.rank))
        self._unitary = None


class VariationalLayer(ParametrizedGate):
    """Layer of one-qubit parametrized gates followed by two-qubit entangling gates.

    Performance is optimized by fusing the variational one-qubit gates with the
    two-qubit entangling gates that follow them and applying a single layer of
    two-qubit gates as 4x4 matrices.

    Args:
        qubits (list): List of one-qubit gate target qubit IDs.
        pairs (list): List of pairs of qubit IDs on which the two qubit gate act.
        one_qubit_gate: Type of one qubit gate to use as the variational gate.
        two_qubit_gate: Type of two qubit gate to use as entangling gate.
        params (list): Variational parameters of one-qubit gates as a list that
            has the same length as ``qubits``. These gates act before the layer
            of entangling gates.
        params2 (list): Variational parameters of one-qubit gates as a list that
            has the same length as ``qubits``. These gates act after the layer
            of entangling gates.
        name (str): Optional name for the gate.
            If ``None`` the name ``"VariationalLayer"`` will be used.

    Example:
        ::

            import numpy as np
            from qibo.models import Circuit
            from qibo import gates
            # generate an array of variational parameters for 8 qubits
            theta = 2 * np.pi * np.random.random(8)

            # define qubit pairs that two qubit gates will act
            pairs = [(i, i + 1) for i in range(0, 7, 2)]
            # map variational parameters to qubit IDs
            theta_map = {i: th for i, th in enumerate(theta}
            # define a circuit of 8 qubits and add the variational layer
            c = Circuit(8)
            c.add(gates.VariationalLayer(pairs, gates.RY, gates.CZ, theta_map))
            # this will create an optimized version of the following circuit
            c2 = Circuit(8)
            c.add((gates.RY(i, th) for i, th in enumerate(theta)))
            c.add((gates.CZ(i, i + 1) for i in range(7)))
    """

    def __init__(self, qubits: List[int], pairs: List[Tuple[int, int]],
                 one_qubit_gate, two_qubit_gate,
                 params: List[float], params2: Optional[List[float]] = None,
                 name: Optional[str] = None):
        super(VariationalLayer, self).__init__()
        self.init_args = [qubits, pairs, one_qubit_gate, two_qubit_gate]
        self.init_kwargs = {"params": params, "params2": params2, "name": name}
        self.name = "VariationalLayer" if name is None else name

        self.target_qubits = tuple(qubits)
        self.params = self._create_params_dict(params)
        self._parameters = list(params)
        if params2 is None:
            self.params2 = {}
        else:
            self.params2 = self._create_params_dict(params2)
            self._parameters.extend(params2)
        self.nparams = len(self.params) + len(self.params2)

        self.pairs = pairs
        targets = set(self.target_qubits)
        two_qubit_targets = set(q for p in pairs for q in p)
        additional_targets = targets - two_qubit_targets
        if not additional_targets:
            self.additional_target = None
        elif len(additional_targets) == 1:
            self.additional_target = additional_targets.pop()
        else:
            raise_error(ValueError, "Variational layer can have at most one additional "
                                    "target for one qubit gates but has {}."
                                    "".format(additional_targets))

        self.one_qubit_gate = one_qubit_gate
        self.two_qubit_gate = two_qubit_gate

        self.is_dagger = False
        self.unitaries = []
        self.additional_unitary = None

    def _create_params_dict(self, params: List[float]) -> Dict[int, float]:
        if len(self.target_qubits) != len(params):
            raise_error(ValueError, "VariationalLayer has {} target qubits but {} "
                                    "parameters were given."
                                    "".format(len(self.target_qubits), len(params)))
        return {q: p for q, p in zip(self.target_qubits, params)}

    def _dagger(self) -> "Gate":
        """"""
        import copy
        varlayer = copy.copy(self)
        varlayer.is_dagger = True
        varlayer.unitaries = [u.dagger() for u in self.unitaries]
        if self.additional_unitary is not None:
            varlayer.additional_unitary = self.additional_unitary.dagger()
        return varlayer

    @property
    def parameter(self) -> List[float]:
        return self._parameters

    @parameter.setter
    def parameter(self, x):
        if self.params2:
            n = len(x) // 2
            self.params = self._create_params_dict(x[:n])
            self.params2 = self._create_params_dict(x[n:])
        else:
            self.params = self._create_params_dict(x)
        self._parameters = x

        matrices, additional_matrix = self._calculate_unitaries()
        for unitary, matrix in zip(self.unitaries, matrices):
            unitary.parameter = matrix
        if additional_matrix is not None:
            self.additional_unitary.parameter = additional_matrix


class Flatten(SpecialGate):
    """Passes an arbitrary state vector in the circuit.

    Args:
        coefficients (list): list of the target state vector components.
            This can also be a tensor supported by the backend.
    """

    def __init__(self, coefficients):
        super(Flatten, self).__init__()
        self.name = "Flatten"
        self.coefficients = coefficients
        self.init_args = [coefficients]


class CallbackGate(SpecialGate):
    """Calculates a :class:`qibo.tensorflow.callbacks.Callback` at a specific point in the circuit.

    This gate performs the callback calulation without affecting the state vector.

    Args:
        callback (:class:`qibo.tensorflow.callbacks.Callback`): Callback object to calculate.
    """

    def __init__(self, callback: "Callback"):
        super(CallbackGate, self).__init__()
        self.name = callback.__class__.__name__
        self.callback = callback
        self.init_args = [callback]

    @Gate.nqubits.setter
    def nqubits(self, n: int):
        Gate.nqubits.fset(self, n) # pylint: disable=no-member
        self.callback.nqubits = n


class KrausChannel(Gate):
    """General channel defined by arbitrary Krauss operators.

    Implements the following transformation:

    .. math::
        \\mathcal{E}(\\rho ) = \\sum _k A_k \\rho A_k^\\dagger

    where A are arbitrary Kraus operators given by the user. Note that Kraus
    operators set should be trace preserving, however this is not checked.
    Simulation of this gate requires the use of density matrices.
    For more information on channels and Kraus operators please check
    `J. Preskill's notes <http://theory.caltech.edu/~preskill/ph219/chap3_15.pdf>`_.

    Args:
        gates (list): List of Kraus operators as pairs ``(qubits, Ak)`` where
          ``qubits`` refers the qubit ids that ``Ak`` acts on and ``Ak`` is
          the corresponding matrix as a ``np.ndarray`` or ``tf.Tensor``.

    Example:
        ::

            from qibo.models import Circuit
            from qibo import gates
            # initialize circuit with 3 qubits
            c = Circuit(3)
            # define a sqrt(0.4) * X gate
            a1 = np.sqrt(0.4) * np.array([[0, 1], [1, 0]])
            # define a sqrt(0.6) * CNOT gate
            a2 = np.sqrt(0.6) * np.array([[1, 0, 0, 0], [0, 1, 0, 0],
                                          [0, 0, 0, 1], [0, 0, 1, 0]])
            # define the channel rho -> 0.4 X{1} rho X{1} + 0.6 CNOT{0, 2} rho CNOT{0, 2}
            channel = gates.GeneralChannel([((1,), a1), ((0, 2), a2)])
            # add the channel to the circuit
            c.add(channel)
    """

    def __init__(self, gates):
        super(KrausChannel, self).__init__()
        self.name = "KrausChannel"
        if isinstance(gates[0], Gate):
            self.gates = tuple(gates)
            self.target_qubits = tuple(sorted(set(
                q for gate in gates for q in gate.target_qubits)))
        else:
            self.gates, self.target_qubits = self._from_matrices(gates)
        self.init_args = [self.gates]

    def _from_matrices(self, matrices):
        """Creates gates from qubits and matrices list."""
        gatelist, qubitset = [], set()
        for qubits, matrix in matrices:
            # Check that given operators have the proper shape.
            rank = 2 ** len(qubits)
            shape = tuple(matrix.shape)
            if shape != (rank, rank):
                raise_error(ValueError, "Invalid Krauss operator shape {} for "
                                        "acting on {} qubits."
                                        "".format(shape, len(qubits)))
            qubitset.update(qubits)
            gatelist.append(self.module.Unitary(matrix, *list(qubits)))
        return tuple(gatelist), tuple(sorted(qubitset))

    def controlled_by(self, *q): # pylint: disable=W0613
        """"""
        raise_error(ValueError, "Noise channel cannot be controlled on qubits.")

    def on_qubits(self, *q): # pragma: no cover
        # future TODO
        raise_error(NotImplementedError, "`on_qubits` method is not available "
                                         "for the `GeneralChannel` gate.")


class UnitaryChannel(KrausChannel):
    """Channel that is a probabilistic sum of unitary operations.

    Implements the following transformation:

    .. math::
        \\mathcal{E}(\\rho ) = \\left (1 - \\sum _k p_k \\right )\\rho +
                                \\sum _k p_k U_k \\rho U_k^\\dagger

    where U are arbitrary unitary operators and p are floats between 0 and 1.
    Note that unlike :class:`qibo.base.gates.KrausChannel` which requires
    density matrices, it is possible to simulate the unitary channel using
    state vectors and probabilistic sampling. For more information on this
    approach we refer to :ref:`Using repeated execution <repeatedexec-example>`.

    Args:
        p (list): List of floats that correspond to the probability that each
            unitary Uk is applied.
        gates (list): List of  operators as pairs ``(qubits, Uk)`` where
            ``qubits`` refers the qubit ids that ``Uk`` acts on and ``Uk`` is
            the corresponding matrix as a ``np.ndarray``/``tf.Tensor``.
            Must have the same length as the given probabilities ``p``.
        seed (int): Optional seed for the random number generator when sampling
            instead of density matrices is used to simulate this gate.
    """

    def __init__(self, p, gates, seed=None):
        if len(p) != len(gates):
            raise_error(ValueError, "Probabilities list has length {} while "
                                    "{} gates were given."
                                    "".format(len(p), len(gates)))
        for pp in p:
            if pp < 0 or pp > 1:
                raise_error(ValueError, "Probabilities should be between 0 "
                                        "and 1 but {} was given.".format(pp))
        super(UnitaryChannel, self).__init__(gates)
        self.name = "UnitaryChannel"
        self.probs = p
        self.psum = sum(p)
        if self.psum > 1 or self.psum <= 0:
            raise_error(ValueError, "UnitaryChannel probability sum should be "
                                    "between 0 and 1 but is {}."
                                    "".format(self.psum))
        self.seed = seed
        self.init_args = [p, self.gates]
        self.init_kwargs = {"seed": seed}


class PauliNoiseChannel(UnitaryChannel):
    """Noise channel that applies Pauli operators with given probabilities.

    Implements the following transformation:

    .. math::
        \\mathcal{E}(\\rho ) = (1 - p_x - p_y - p_z) \\rho + p_x X\\rho X + p_y Y\\rho Y + p_z Z\\rho Z

    which can be used to simulate phase flip and bit flip errors.
    This channel can be simulated using either density matrices or state vectors
    and sampling with repeated execution.
    See :ref:`How to perform noisy simulation? <noisy-example>` for more
    information.

    Args:
        q (int): Qubit id that the noise acts on.
        px (float): Bit flip (X) error probability.
        py (float): Y-error probability.
        pz (float): Phase flip (Z) error probability.
        seed (int): Optional seed for the random number generator when sampling
            instead of density matrices is used to simulate this gate.
    """

    def __init__(self, q, px=0, py=0, pz=0, seed=None):
        probs, gates = [], []
        for p, gate in [(px, "X"), (py, "Y"), (pz, "Z")]:
            if p > 0:
                probs.append(p)
                gates.append(getattr(self.module, gate)(q))

        super(PauliNoiseChannel, self).__init__(probs, gates, seed=seed)
        self.name = "PauliNoiseChannel"
        assert self.target_qubits == (q,)

        self.init_args = [q]
        self.init_kwargs = {"px": px, "py": py, "pz": pz, "seed": seed}


class ResetChannel(UnitaryChannel):
    r"""Single-qubit reset channel.

    Implements the following transformation:

    .. math::
        \\mathcal{E}(\\rho ) = (1 - p_0 - p_1) \\rho
        + p_0 (|0\\rangle \\langle 0| \\otimes \\tilde{\\rho })
        + p_1 (|1\\rangle \langle 1| \otimes \\tilde{\\rho })

    with

    .. math::
        \\tilde{\\rho } = \\frac{\langle 0|\\rho |0\\rangle }{\mathrm{Tr}\langle 0|\\rho |0\\rangle}

    using :class:`qibo.base.gates.Collapse`.

    Args:
        q (int): Qubit id that the channel acts on.
        p0 (float): Probability to reset to 0.
        p1 (float): Probability to reset to 1.
        seed (int): Optional seed for the random number generator when sampling
            instead of density matrices is used to simulate this gate.
    """

    def __init__(self, q, p0=0.0, p1=0.0, seed=None):
        probs = [p0, p1]
        gates = [self.module.Collapse(q), self.module.X(q)]
        super(ResetChannel, self).__init__(probs, gates, seed=seed)
        self.name = "ResetChannel"
        assert self.target_qubits == (q,)

        self.init_args = [q]
        self.init_kwargs = {"p0": p0, "p1": p1, "seed": seed}


class ThermalRelaxationChannel:
    r"""Single-qubit thermal relaxation error channel.

    Implements the following transformation:

    If :math:`T_1 \\geq T_2`:

    .. math::
        \\mathcal{E} (\\rho ) = (1 - p_z - p_0 - p_1)\\rho + p_zZ\\rho Z
        + p_0 (|0\\rangle \\langle 0| \\otimes \\tilde{\\rho })
        + p_1 (|1\\rangle \langle 1| \otimes \\tilde{\\rho })

    with

    .. math::
        \\tilde{\\rho } = \\frac{\langle 0|\\rho |0\\rangle }{\mathrm{Tr}\langle 0|\\rho |0\\rangle}

    while if :math:`T_1 < T_2`:

    .. math::
        \\mathcal{E}(\\rho ) = \\mathrm{Tr} _\\mathcal{X}\\left [\\Lambda _{\\mathcal{X}\\mathcal{Y}}(\\rho _\\mathcal{X} ^T \\otimes \\mathbb{I}_\\mathcal{Y})\\right ]

    with

    .. math::
        \\Lambda = \\begin{pmatrix}
        1 - p_1 & 0 & 0 & e^{-t / T_2} \\\\
        0 & p_1 & 0 & 0 \\\\
        0 & 0 & p_0 & 0 \\\\
        e^{-t / T_2} & 0 & 0 & 1 - p_0
        \\end{pmatrix}

    where :math:`p_0 = (1 - e^{-t / T_1})(1 - \\eta )` :math:`p_1 = (1 - e^{-t / T_1})\\eta`
    and :math:`p_z = 1 - e^{-t / T_1} + e^{-t / T_2} - e^{t / T_1 - t / T_2}`.
    Here :math:`\\eta` is the ``excited_population``
    and :math:`t` is the ``time``, both controlled by the user.
    This gate is based on
    `Qiskit's thermal relaxation error channel <https://qiskit.org/documentation/stubs/qiskit.providers.aer.noise.thermal_relaxation_error.html#qiskit.providers.aer.noise.thermal_relaxation_error>`_.

    Args:
        q (int): Qubit id that the noise channel acts on.
        t1 (float): T1 relaxation time. Should satisfy ``t1 > 0``.
        t2 (float): T2 dephasing time.
            Should satisfy ``t1 > 0`` and ``t2 < 2 * t1``.
        time (float): the gate time for relaxation error.
        excited_population (float): the population of the excited state at
            equilibrium. Default is 0.
        seed (int): Optional seed for the random number generator when sampling
            instead of density matrices is used to simulate this gate.
    """

    def __init__(self, q, t1, t2, time, excited_population=0, seed=None):
        self.name = "ThermalRelaxationChannel"
        self.init_args = [q, t1, t2, time]
        self.init_kwargs = {"excited_population": excited_population,
                            "seed": seed}

    @staticmethod
    def _calculate_probs(t1, t2, time, excited_population):
        if excited_population < 0 or excited_population > 1:
            raise_error(ValueError, "Invalid excited state population {}."
                                    "".format(excited_population))
        if time < 0:
            raise_error(ValueError, "Invalid gate_time ({} < 0)".format(time))
        if t1 <= 0:
            raise_error(ValueError, "Invalid T_1 relaxation time parameter: "
                                    "T_1 <= 0.")
        if t2 <= 0:
            raise_error(ValueError, "Invalid T_2 relaxation time parameter: "
                                    "T_2 <= 0.")
        if t2 > 2 * t1:
            raise_error(ValueError, "Invalid T_2 relaxation time parameter: "
                                    "T_2 greater than 2 * T_1.")

        p_reset = 1 - np.exp(-time / t1)
        p0 = p_reset * (1 - excited_population)
        p1 = p_reset * excited_population
        if t1 < t2:
            exp = np.exp(-time / t2)
        else:
            rate1, rate2 = 1 / t1, 1 / t2
            exp = (1 - p_reset) * (1 - np.exp(-time * (rate2 - rate1))) / 2
        return (exp, p0, p1)


class _ThermalRelaxationChannelA(UnitaryChannel):
    """Implements thermal relaxation when T1 >= T2."""

    def __init__(self, q, t1, t2, time, excited_population=0, seed=None):
        probs = ThermalRelaxationChannel._calculate_probs(
            t1, t2, time, excited_population)
        gates = [self.module.Z(q), self.module.Collapse(q),
                 self.module.X(q)]

        super(_ThermalRelaxationChannelA, self).__init__(
            probs, gates, seed=seed)
        ThermalRelaxationChannel.__init__(
            self, q, t1, t2, time, excited_population=excited_population,
            seed=seed)
        assert self.target_qubits == (q,)


class _ThermalRelaxationChannelB(Gate):
    """Implements thermal relaxation when T1 < T2."""

    def __init__(self, q, t1, t2, time, excited_population=0, seed=None):
        probs = ThermalRelaxationChannel._calculate_probs(
            t1, t2, time, excited_population)
        self.exp_t2, self.preset0, self.preset1 = probs

        super(_ThermalRelaxationChannelB, self).__init__()
        self.target_qubits = (q,)
        ThermalRelaxationChannel.__init__(
            self, q, t1, t2, time, excited_population=excited_population,
            seed=seed)
