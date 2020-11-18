from qibo.config import BACKEND_NAME, raise_error
if BACKEND_NAME == "tensorflow":
    from qibo.tensorflow.callbacks import ( # pylint: disable=W0611
        Callback, PartialTrace, EntanglementEntropy, Norm, Overlap,
        Energy, Gap
        )
else: # pragma: no cover
    # case not tested because backend is preset to TensorFlow
    raise_error(NotImplementedError, "Only Tensorflow backend is implemented.")
