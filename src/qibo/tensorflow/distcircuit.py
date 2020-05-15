# -*- coding: utf-8 -*-
# @authors: S. Efthymiou
import copy
import numpy as np
import tensorflow as tf
import joblib
from qibo.config import DTYPECPX, DTYPEINT
from qibo.tensorflow import circuit, gates, matrices, measurements, callbacks
from typing import Dict, List, Optional, Sequence, Tuple, Union

# TODO: Fix device matrices and `TensorflowGate._construct_matrix`
# TODO: Implement swap

class TensorflowDistributedCircuit(circuit.TensorflowCircuit):
    """Implementation of :class:`qibo.base.circuit.BaseCircuit` in Tensorflow.

    Args:
        nqubits (int): Total number of qubits in the circuit.
        dtype: Tensorflow type for complex numbers.
            Read automatically from `config`.
    """
    _GATE_MODULE = gates

    def __init__(self,
                 nqubits: int,
                 calc_devices: Dict[str, int],
                 memory_device: str = "/CPU:0",
                 dtype=DTYPECPX):
        super(TensorflowDistributedCircuit, self).__init__(nqubits, dtype)

        self.ndevices = sum(calc_devices.values())
        self.nglobal = np.log2(self.ndevices)
        if not self.nglobal.is_integer():
            raise ValueError("Number of calculation devices should be a power "
                             "of 2 but is {}.".format(self.ndevices))
        self.nglobal = int(self.nglobal)

        self.memory_device = memory_device
        self.calc_devices = calc_devices

        self.queues = {d: [] for d in self.calc_devices.keys()}
        self.global_qubits_list = []

        # Construct gate matrices casted in each calculation device
        self.matrices = {}
        for device in self.calc_devices.keys():
            with tf.device(device):
                self.matrices[device] = matrices.GateMatrices(self.dtype)

        self.pieces = None
        self._global_qubits = None
        self._local_qubits = None

    @property
    def global_qubits(self) -> List[int]:
        if self._global_qubits is None:
            raise ValueError("Cannot access global qubits before being set.")
        return self._global_qubits

    @global_qubits.setter
    def global_qubits(self, x: Sequence[int]):
        if len(x) != self.nglobal:
            raise ValueError("Invalid number of global qubits {} for using {} "
                             "calculation devices.".format(len(x), self.ndevices))
        self._global_qubits = set(x)
        self._local_qubits = [i for i in range(self.nqubits) if i not in self._global_qubits]

        self.transpose_order = list(sorted(self._global_qubits)) + self._local_qubits
        self.reverse_transpose_order = self.nqubits * [0]
        for i, v in enumerate(self.transpose_order):
            self.reverse_transpose_order[v] = i

    def __add__(self, circuit: "TensorflowCircuit") -> "TensorflowCircuit":
        return TensorflowCircuit._circuit_addition(self, circuit)

    def _set_nqubits(self, gate):
        # Do not set ``gate.nqubits`` during gate addition because this will
        # be set by the ``_set_gates`` method once all gates are known.
        pass

    def _set_gates(self):
        all_qubits = set(range(self.nqubits))
        queues = [[]]

        global_qubits = set(all_qubits)
        queue = iter(self.queue)
        try:
            gate = next(queue)
            while True:
                target_qubits = set(gate.qubits)
                global_qubits -= target_qubits
                while len(global_qubits) > self.nglobal:
                    queues[-1].append(gate)
                    gate = next(queue)
                    target_qubits = set(gate.qubits)
                    global_qubits -= target_qubits

                if len(global_qubits) == self.nglobal:
                    queues[-1].append(gate)
                    gate = next(queue)
                else:
                    # must be len(global_qubits) < self.nglobal
                    free_qubits = list(sorted(target_qubits))
                    global_qubits |= set(free_qubits[self.nglobal - len(global_qubits):])

                queues.append([])
                self.global_qubits_list.append(list(sorted(global_qubits)))
                global_qubits = set(all_qubits)

        except StopIteration:
            if len(global_qubits) > self.nglobal:
                global_qubits = list(sorted(global_qubits))[:self.nglobal]
            self.global_qubits_list.append(list(sorted(global_qubits)))

        # "Compile" actual gates
        nlocal = self.nqubits - self.nglobal
        for global_qubits, queue in zip(self.global_qubits_list, queues):
            for device in self.calc_devices.keys():
                self.queues[device].append([])

            for gate in queue:
                gate.no_matrix = True
                gate.reduce(global_qubits)
                gate.nqubits = nlocal
                for device in self.calc_devices.keys():
                    calc_gate = copy.copy(gate)
                    with tf.device(device):
                        calc_gate.matrices = self.matrices[device]
                        calc_gate._construct_matrix()
                    self.queues[device][-1].append(calc_gate)

    def compile(self, callback: Optional[callbacks.Callback] = None):
        """Compiles the circuit as a Tensorflow graph.

        Args:
            callback: A Callback to calculate during circuit execution.
                See :class:`qibo.tensorflow.callbacks.Callback` for more details.
                User can give a single callback or list of callbacks here.
        """
        raise NotImplementedError("Compiling is not implemented for "
                                  "distributed circuits yet.")

    @staticmethod
    def _device_execute_state(state: tf.Tensor,
                              gates: List[gates.TensorflowGate]) -> tf.Tensor:
        for gate in gates:
            state = gate(state)
        return state

    def _cast_results(self, results: List[List[tf.Tensor]]):
        i = 0
        for result in results:
            for s in result:
                self.pieces[i].assign(s)
                i += 1

    def _joblib_config(self) -> Tuple[List[tf.Tensor], str]:
        start = 0
        for device, n in self.calc_devices.items():
            stop = start + n
            yield self.pieces[start: stop], device
            start = stop

    def execute(self,
                initial_state: Optional[Union[np.ndarray, tf.Tensor]] = None,
                nshots: Optional[int] = None,
                callback: Optional[callbacks.Callback] = None
                ) -> Union[tf.Tensor, measurements.CircuitResult]:
        """Propagates the state through the circuit applying the corresponding gates.

        In default usage the full final state vector or density matrix is returned.
        If the circuit contains measurement gates and `nshots` is given, then
        the final state is sampled and the samples are returned.
        Circuit execution uses by default state vectors but switches automatically
        to density matrices if

        Args:
            initial_state (np.ndarray): Initial state vector as a numpy array of shape ``(2 ** nqubits,)``
                or a density matrix of shape ``(2 ** nqubits, 2 ** nqubits)``.
                A Tensorflow tensor with shape ``nqubits * (2,)`` (or ``2 * nqubits * (2,)`` for density matrices)
                is also allowed as an initial state but must have the `dtype` of the circuit.
                If ``initial_state`` is ``None`` the |000...0> state will be used.
            nshots (int): Number of shots to sample if the circuit contains
                measurement gates.
                If ``nshots`` None the measurement gates will be ignored.
            callback: A Callback to calculate during circuit execution.
                See :class:`qibo.tensorflow.callbacks.Callback` for more details.
                User can give a single callback or list of callbacks here.
                Note that if the Circuit is compiled then all callbacks should
                be passed when ``compile`` is called, not during execution.
                Otherwise an ``RuntimeError`` will be raised.

        Returns:
            If ``nshots`` is given and the circuit contains measurements
                A :class:`qibo.base.measurements.CircuitResult` object that contains the measured bitstrings.
            If ``nshots`` is ``None`` or the circuit does not contain measurements.
                The final state vector as a Tensorflow tensor of shape ``(2 ** nqubits,)`` or a density matrix of shape ``(2 ** nqubits, 2 ** nqubits)``.
        """
        self.global_qubits = self.global_qubits_list[0]
        if initial_state is None:
            self._set_initial_state(initital_state)

        if self.compiled_execute is None:
            self._add_callbacks(callback)

            for group, global_qubits in enumerate(self.global_qubits_list):
                if group > 0:
                    self._swap(global_qubits)

                def _device_job(states, device):
                    with tf.device(device):
                        return [self._device_execute_state(s, self.queues[device][group])
                                for s in states]

                pool = joblib.Parallel(n_jobs=len(self.calc_devices),
                                       prefer="threads")
                results = pool(joblib.delayed(gpu_job)(s, d)
                               for s, d in self._joblib_config())
                self._cast_results(results)

        else:
            if callback is not None:
                raise RuntimeError("Cannot add callbacks to compiled circuit. "
                                   "Please pass the callbacks when compiling.")
            state, callback_results = self.compiled_execute(state)

        # Append callback results to callbacks
        for callback, result in zip(self.callbacks, callback_results):
            callback.append(result)

        if self.measurement_gate is None or nshots is None:
            return self.final_state

        raise NotImplementedError("Measurements are not implemented for "
                                  "distributed circuits.")
        #samples = self.measurement_gate(state, nshots, samples_only=True,
        #                                is_density_matrix=self.using_density_matrix)
        #self._final_state = state

        #self.measurement_gate_result = measurements.GateResult(
        #    self.measurement_gate.qubits, state, decimal_samples=samples)
        #return measurements.CircuitResult(
        #    self.measurement_tuples, self.measurement_gate_result)

    @property
    def final_state(self) -> tf.Tensor:
        """Final state as a Tensorflow tensor of shape (2 ** nqubits,).

        The circuit has to be executed at least once before accessing this
        property, otherwise a `ValueError` is raised. If the circuit is
        executed more than once, only the last final state is returned.
        """
        if self.pieces is None:
            raise ValueError("Cannot access the state tensor before being set.")
        return self._merge(self.pieces)

    def _default_global_qubits(self) -> List[int]:
        """Returns a list with the last qubits to cast them as global."""
        return list(range(self.nqubits - self.nglobal, self.nqubits))

    def _default_initial_state(self) -> tf.Tensor:
        """Creates the |000...0> state for default initialization."""
        if self.pieces is None:
            raise RuntimeError("Attempting to initialize distributed circuit "
                               "state that is already initialized.")

        if self._global_qubits is None:
            self.global_qubits = self._default_global_qubits()

        # Generate initial state piece
        n = self.nqubits - self.nglobal
        _state = tf.zeros(2 ** n, dtype=self.dtype)
        update = tf.constant([1], dtype=self.dtype)
        _state = tf.tensor_scatter_nd_update(initial_state,
                                             tf.constant([[0]], dtype=DTYPEINT),
                                             update)
        _state = tf.reshape(_state, n * (2,))
        # Cast initial state piece to all variable pieces
        with tf.device(self.memory_device):
            self.pieces = [tf.Variable(_state) for _ in range(self.ndevices)]

    def _set_initial_state(self, initial_state: Optional[Union[np.ndarray, tf.Tensor]] = None) -> tf.Tensor:
        """Checks and casts initial state given by user."""
        if initial_state is None:
            return self._default_initial_state()

        state = super(TensorflowDistributedCircuit, self)._set_initial_state(initial_state)
        # TODO: Cast `state` (tf.Tensor) to `self.pieces`
        raise NotImplementedError("User given initial state is not implemented "
                                  "for distributed circuits yet.")

    def _add_callbacks(self, callback: callbacks.Callback):
        """Adds callbacks in the circuit."""
        raise NotImplementedError("Callbacks are not implemented for "
                                  "distributed circuits.")
        #n = len(self.callbacks)
        #if isinstance(callback, list):
        #    self.callbacks += callback
        #elif isinstance(callback, callbacks.Callback):
        #    self.callbacks.append(callback)
        # Set number of qubits in new callbacks
        #for cb in self.callbacks[n:]:
        #    cb.nqubits = self.nqubits

    def _split(self, state: tf.Tensor):
        shape = (self.ndevices,) + (self.nqubits - self.nglobal) * (2,)
        with tf.device(self.memory_device):
            state = tf.transpose(state, self.transpose_order)
            state = tf.reshape(state, shape)
            for i in range(self.ndevices):
                self.pieces[i].assign(state[i])

    def _merge(self, states: List[tf.Tensor]) -> tf.Tensor:
        with tf.device(self.memory_device):
            state = tf.concat([s[tf.newaxis] for s in states], axis=0)
            state = tf.reshape(state, self.nqubits * (2,))
            state = tf.transpose(state, self.reverse_transpose_order)
            return state

    def _swap(self, new_global_qubits: Sequence[int]):
        # TODO: Fix this
        raise NotImplementedError
