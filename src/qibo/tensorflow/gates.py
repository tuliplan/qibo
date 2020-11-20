# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
from qibo.config import get_device, raise_error, DTYPES
from abc import ABC, abstractmethod


class BaseTensorflowGate(ABC):

    def __init__(self):
        self._unitary = None
        self.is_prepared = False
        # Cast gate matrices to the proper device
        self.device = get_device()
        # Reference to copies of this gate that are casted in devices when
        # a distributed circuit is used
        self.device_gates = set()
        self.original_gate = None
        # Using density matrices or state vectors
        self._density_matrix = False
        self._active_call = "_state_vector_call"

    @property
    def density_matrix(self) -> bool:
        return self._density_matrix

    @density_matrix.setter
    def density_matrix(self, x: bool):
        if self._nqubits is not None:
            raise_error(RuntimeError,
                        "Density matrix mode cannot be switched after "
                        "preparing the gate for execution.")
        self._density_matrix = x
        if x:
            self._active_call = "_density_matrix_call"
        else:
            self._active_call = "_state_vector_call"

    @abstractmethod
    def _construct_unitary(self):
        raise_error(NotImplementedError)

    @staticmethod
    def _control_unitary(unitary):
        shape = tuple(unitary.shape)
        if shape != (2, 2):
            raise_error(ValueError, "Cannot control unitary matrix of "
                                    "shape {}.".format(shape))
        dtype = DTYPES.get('DTYPECPX')
        zeros = tf.zeros((2, 2), dtype=dtype)
        part1 = tf.concat([tf.eye(2, dtype=dtype), zeros], axis=0)
        part2 = tf.concat([zeros, unitary], axis=0)
        return tf.concat([part1, part2], axis=1)

    def construct_unitary(self):
        unitary = self._construct_unitary()
        if self.is_controlled_by:
            return self._control_unitary(unitary)
        return unitary

    @abstractmethod
    def _state_vector_call(self, state): # pragma: no cover
        """Acts with the gate on a given state vector."""
        raise_error(NotImplementedError)

    @abstractmethod
    def _density_matrix_call(self, state): # pragma: no cover
        """Acts with the gate on a given density matrix."""
        raise_error(NotImplementedError)

    @abstractmethod
    def _nqubits_from_state(self, state): # pragma: no cover
        """Sets the total number of qubits of the state that the gate acts on."""
        raise_error(NotImplementedError)

    @abstractmethod
    def prepare(self): # pragma: no cover
        raise_error(NotImplementedError)

    def __call__(self, state):
        """Acts with the gate on a given state vector or density matrix.

        Args:
            state: Input state vector.
                The type and shape of this depend on the backend.

        Returns:
            The state vector after the action of the gate.
        """
        if self._nqubits is None:
            self.nqubits = self._nqubits_from_state(state)
        if not self.is_prepared:
            self.prepare()
        return getattr(self, self._active_call)(state)
