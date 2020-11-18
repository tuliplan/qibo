# -*- coding: utf-8 -*-
# @authors: S. Efthymiou
import collections
from abc import ABC, abstractmethod
from typing import Any, Optional, Dict, List, Set, Tuple, Union
from qibo.config import raise_error
TensorType = Any
ProbsType = Union[float, List[float], Dict[int, float]]


class GateResult(ABC):
    """Object returned when user uses a `gates.M` on a state.

    Implements tools to convert samples from decimal to binary representation
    (and vice versa) and calculating the frequencies of shots.

    Args:
        qubits: Sorted tuple of qubit ids that the measurement gate acts on.
        decimal_samples: Tensor holding the measured samples in decimal
            representation. Has shape (nshots,).
        binary_samples: Tensor holding the measured samples in binary
            representation. Has shape (nshots, len(qubits)).
        Exactly one of `decimal_samples`, `binary_samples` should be given to
        create the object.
    """

    def __init__(self, qubits: Tuple[int],
                 decimal_samples: Optional[TensorType] = None,
                 binary_samples: Optional[TensorType] = None):
        self.qubits = qubits

        if decimal_samples is not None and binary_samples is not None:
            raise_error(ValueError, "Measurement result object cannot be created "
                                    "when samples are given both in decimal and "
                                    "binary. Use one of the two.")
        if binary_samples is not None and binary_samples.shape[-1] != self.nqubits:
            raise_error(ValueError, "Binary samples are for {} qubits but the given "
                                    "qubits are {}.".format(binary_samples.shape[-1], qubits))

        self._decimal = decimal_samples
        self._binary = binary_samples
        self._frequencies = None

    @property
    def nqubits(self) -> int:
        return len(self.qubits)

    @property
    def qubit_map(self) -> Dict[int, int]:
        return {q: i for i, q in enumerate(self.qubits)}

    def samples(self, binary: bool = True) -> TensorType:
        if binary:
            if self._binary is None:
                self._binary = self._convert_to_binary(
                    self._decimal, self.nqubits)
            return self._binary

        if self._decimal is None:
            self._decimal = self._convert_to_decimal(self._binary, self.nqubits)
        return self._decimal

    def __getitem__(self, i: int) -> TensorType:
        return self.samples(binary=False)[i]

    def frequencies(self, binary: bool = True) -> collections.Counter:
        """Calculates frequencies of appearance of each measurement.

        Args:
            binary: If True the returned keys (measurement results) are in
                binary representation. Otherwise they are in decimal
                representation.

        Returns:
            A `collections.Counter` where the keys are the measurement results
            and the values are the number of appearances of each result in the
            measured shots.
        """
        if self._frequencies is None:
            res, cnts = self._calculate_counts(self.samples(binary=False))
            self._frequencies = collections.Counter(
                {k: v for k, v in zip(res, cnts)})
        if binary:
            return collections.Counter(
                {"{0:b}".format(k).zfill(self.nqubits): v
                 for k, v in self._frequencies.items()})
        return self._frequencies

    @staticmethod
    def _get_bitflip_tuple(qubits: Tuple[int], probs: ProbsType) -> Tuple[float]:
        if isinstance(probs, float):
            if probs < 0 or probs > 1:
                raise_error(ValueError, "Invalid bitflip probability {}."
                                        "".format(probs))
            return len(qubits) * (probs,)

        if isinstance(probs, (tuple, list)):
            if len(probs) != len(qubits):
                raise_error(ValueError, "{} qubits were measured but the given "
                                        "bitflip probability list contains {} "
                                        "values.".format(
                                            len(qubits), len(probs)))
            return tuple(probs)

        if isinstance(probs, dict):
            diff = set(probs.keys()) - set(qubits)
            if diff:
                raise_error(KeyError, "Bitflip map contains {} qubits that are "
                                      "not measured.".format(diff))
            return tuple(probs[q] if q in probs else 0.0 for q in qubits)

        raise_error(TypeError, "Invalid type {} of bitflip map.".format(probs))

    def apply_bitflips(self, p0: ProbsType, p1: Optional[ProbsType] = None
                       ) -> "GateResult":
        """Applies bitflip noise to the measured samples.

        Args:
            p0: Bitflip probability map. Can be:
                A dictionary that maps each measured qubit to the probability
                that it is flipped, a list or tuple that has the same length
                as the tuple of measured qubits or a single float number.
                If a single float is given the same probability will be used
                for all qubits.
            p1: Probability of asymmetric bitflip. If ``p1`` is given, ``p0``
                will be used as the probability for 0->1 and ``p1`` as the
                probability for 1->0. If ``p1`` is ``None`` the same probability
                ``p0`` will be used for both bitflips.

        Returns:
            A new :class:`qibo.base.measurements.GateResult` object that holds
            the noisy samples.
        """
        if p1 is None:
            probs = 2 * (self._get_bitflip_tuple(self.qubits, p0),)
        else:
            probs = (self._get_bitflip_tuple(self.qubits, p0),
                     self._get_bitflip_tuple(self.qubits, p1))
        noisy = self._apply_bitflips(self.samples(), probs)
        return self.__class__(self.qubits, binary_samples=noisy)

    @staticmethod
    @abstractmethod
    def _convert_to_binary(x: TensorType, n: int) -> TensorType: # pragma: no cover
        raise_error(NotImplementedError)

    @staticmethod
    @abstractmethod
    def _convert_to_decimal(x: TensorType, n: int) -> TensorType: # pragma: no cover
        raise_error(NotImplementedError)

    @staticmethod
    @abstractmethod
    def _calculate_counts(decimal_samples: TensorType) -> Tuple[List[int]]: # pragma: no cover
        raise_error(NotImplementedError)

    @staticmethod
    @abstractmethod
    def _apply_bitflips(noiselss_samples: TensorType, probs: Dict[int, float]
                        ) -> TensorType: # pragma: no cover
        raise_error(NotImplementedError)


class CircuitResult:
    """Object returned when user performs measurements using a circuit.

    Implements tools for dividing the global measurements from the circuit's
    `measurement_gate` to the corresponding registers.
    This object is created automatically every time a circuit that contains
    measurement gates is executed. The user does not have to worry about
    creating this object.

    Args:
        register_qubits: Dictionary that maps register names to the
            corresponding tuples of qubit ids. This is created in the
            `measurement_tuples` variable of :class:`qibo.base.circuit.BaseCircuit`.
        measurement_gate_result: The `GateResult` resulting from the circuit's
            global measurement gate.
    """

    def __init__(self,
                 register_qubits: Dict[str, Tuple[int]],
                 measurement_gate_result: GateResult):
        self.register_qubits = register_qubits
        self.result = measurement_gate_result
        self.__register_results = None

    @property
    def _register_results(self) -> Dict[str, GateResult]:
        """Returns the individual `GateResult`s for each register."""
        if self.__register_results is None:
            self.__register_results = self._calculate_register_results(
                self.register_qubits, self.result)
        return self.__register_results

    def samples(self, binary: bool = True, registers: bool = False
                ) -> Union[TensorType, Dict[str, TensorType]]:
        """Returns raw measurement samples.

        Args:
            binary (bool): Return samples in binary or decimal form.
            registers (bool): Group samples according to registers.

        Returns:
            If `binary` is `True`
                samples are returned in binary form as a tensor
                of shape `(nshots, n_measured_qubits)`.
            If `binary` is `False`
                samples are returned in decimal form as a tensor
                of shape `(nshots,)`.
            If `registers` is `True`
                samples are returned in a `dict` where the keys are the register
                names and the values are the samples tensors for each register.
            If `registers` is `False`
                a single tensor is returned which contains samples from all the
                measured qubits, independently of their registers.
        """
        if not registers:
            return self.result.samples(binary)
        return {k: v.samples(binary) for k, v in self._register_results.items()}

    def frequencies(self, binary: bool = True, registers: bool = False
                    ) -> Union[collections.Counter, Dict[str, collections.Counter]]:
        """Returns the frequencies of measured samples.

        Args:
            binary (bool): Return frequency keys in binary or decimal form.
            registers (bool): Group frequencies according to registers.

        Returns:
            A `collections.Counter` where the keys are the observed values
            and the values the corresponding frequencies, that is the number
            of times each measured value/bitstring appears.

            If `binary` is `True`
                the keys of the `Counter` are in binary form, as strings of
                0s and 1s.
            If `binary` is `False`
                the keys of the `Counter` are integers.
            If `registers` is `True`
                a `dict` of `Counter` s is returned where keys are the name of
                each register.
            If `registers` is `False`
                a single `Counter` is returned which contains samples from all
                the measured qubits, independently of their registers.
        """
        if not registers:
            return self.result.frequencies(binary)
        return {k: v.frequencies(binary) for k, v in self._register_results.items()}

    def apply_bitflips(self, p0: ProbsType, p1: Optional[ProbsType] = None
                       ) -> "CircuitResult":
        """Applies bitflip noise to the measured samples.

        Args:
            p0: Bitflip probability map. Can be:
                A dictionary that maps each measured qubit to the probability
                that it is flipped, a list or tuple that has the same length
                as the tuple of measured qubits or a single float number.
                If a single float is given the same probability will be used
                for all qubits.
            p1: Probability of asymmetric bitflip. If ``p1`` is given, ``p0``
                will be used as the probability for 0->1 and ``p1`` as the
                probability for 1->0. If ``p1`` is ``None`` the same probability
                ``p0`` will be used for both bitflips.

        Returns:
            A new :class:`qibo.base.measurements.CircuitResult` object that
            holds the noisy samples.
        """
        noisy_result = self.result.apply_bitflips(p0, p1)
        return self.__class__(self.register_qubits, noisy_result)

    @staticmethod
    @abstractmethod
    def _calculate_register_results(register_qubits: Dict[str, Set[int]],
                                    gate_result: GateResult
                                    ) -> Dict[str, GateResult]: # pragma: no cover
        """Calculates the individual register `GateResults`.

        This uses the `register_qubits` map to divide the bitstrings to their
        appropriate registers.
        """
        raise_error(NotImplementedError)
