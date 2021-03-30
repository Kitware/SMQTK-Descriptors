Descriptor Generation
=====================
We provide the interface :class:`.DescriptorGenerator` to define the high-level
behavior for transforming input blob data, in the form of a
:class:`smqtk_dataprovider.DataElement` [#]_, into a descriptor (feature vector).

This interface also descends from the
:class:`smqtk_dataprovider.ContentTypeValidator` [#]_ interface to allow
implementations the ability to declare what input data content types it can
accept for processing.
Thus, input :class:`.DataElement` instances must be of a content type that the
:class:`.DescriptorGenerator` supports, otherwise an exception is raised when
the offending data element is reached.

Descriptors may be generated most simply as :class:`numpy.ndarray` arrays via
the :meth:`.DescriptorGenerator.generate_arrays`.
An additional layer of wrapping into a :class:`.DescriptorElement` may be
invoked via :meth:`.DescriptorGenerator.generate_elements`.

.. [#] TODO: fill in appropriate link to DataElement interface under
  https://smqtk-dataprovider.readthedocs.io/
.. [#] TODO: fill in appropriate link to ContentTypeValidator interface under
  https://smqtk-dataprovider.readthedocs.io/

Bundled Implementation Model Details
------------------------------------
The :class:`.DescriptorGenerator` interface does not define a model building
method, but some implementations require internal models.
Below are explanations on how to build or get modes for
:class:`.DescriptorGenerator` implementations that require a model.

.. _caffe1-models:

Caffe 1.0 Default Image Net
^^^^^^^^^^^^^^^^^^^^^^^^^^^
The :class:`.CaffeDescriptorGenerator`
implementation does not come with a method of training its own models, but requires model files provided by Caffe:
the network model file and the image mean binary protobuf file.

The Caffe source tree provides two scripts to download the specific files (relative to the caffe source tree):

.. code-block:: bash

    # Downloads the network model file
    scripts/download_model_binary.py models/bvlc_reference_caffenet

    # Downloads the ImageNet mean image binary protobuf file
    data/ilsvrc12/get_ilsvrc_aux.sh

These script effectively just download files from a specific source.

If the Caffe source tree is not available, the model files can be downloaded from the following URLs:

    - Network model: http://dl.caffe.berkeleyvision.org/bvlc_reference_caffenet.caffemodel
    - Image mean: http://dl.caffe.berkeleyvision.org/caffe_ilsvrc12.tar.gz


Reference
---------
.. autoclass:: smqtk_descriptors.interfaces.descriptor_generator.DescriptorGenerator
   :members:

.. autoclass:: smqtk_descriptors.impls.descriptor_generator.caffe1.CaffeDescriptorGenerator
   :members:
