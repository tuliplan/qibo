from qibo.config import BACKEND_NAME, raise_error
if BACKEND_NAME == "tensorflow":
    from qibo.tensorflow.cgates import ( # pylint: disable=W0611
        H, X, Y, Z, I, Collapse, M, RX, RY, RZ, U1, U2, U3, ZPow,
        CNOT, CZ, CRX, CRY, CRZ, CU1, CU2, CU3, CZPow, SWAP, fSim, TOFFOLI,
        GeneralizedfSim, Unitary, VariationalLayer, Flatten, CallbackGate,
        KrausChannel, UnitaryChannel, PauliNoiseChannel, ResetChannel,
        ThermalRelaxationChannel)
else: # pragma: no cover
    # case not tested because backend is preset to TensorFlow
    raise_error(NotImplementedError, "Only Tensorflow backend is implemented.")
