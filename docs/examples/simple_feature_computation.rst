Simple Feature Computation with Caffe 1.0
-----------------------------------------

The following is a concrete example of performing feature computation for a set
of ten butterfly images using the *Caffe 1.0* descriptor generator.
This assumes that you have installed Caffe 1.0's python bindings appropriately
and have downloaded the appropriate model files as detailed in the
:ref:`caffe1-models` section.
Once set up, the following code will compute an *AlexNet* descriptor:

.. code-block:: python

    # Import some butterfly data
    # TODO: This URL is broken. Fix or find alternative example data.
    urls = ["http://www.comp.leeds.ac.uk/scs6jwks/dataset/leedsbutterfly/examples/{:03d}.jpg".format(i) for i in range(1,11)]
    from smqtk.representation.data_element.url_element import DataUrlElement
    el = [DataUrlElement(d) for d in urls]

    # Create an algorithm instance.
    from smqtk_descriptors.impls.descriptor_generator.caffe1 import CaffeDescriptorGenerator
    from smqtk_dataprovider.impls.data_element.file import DataFileElement
    descr_generator = CaffeDescriptorGenerator(
      network_prototxt=DataFileElement("models/bvlc_reference_caffenet/deploy.prototxt"),
      network_model=DataFileElement("models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel"),
      image_mean=DataFileElement("data/ilsvrc12/imagenet_mean.binaryproto"),
    )

    # Compute features on the first image
    result = descr_generator.generate_one_array(el[0])
    print(result)
    # array([ 0.        ,  0.01254855,  0.        , ...,  0.0035853 ,
    #         0.        ,  0.00388408])
