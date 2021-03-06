from io import BytesIO
import itertools
import logging
from typing import Any, Dict, Hashable, Iterable, Mapping, Optional, Set, Tuple, Type, TypeVar

import numpy

from smqtk_core.configuration import (
    from_config_dict,
    to_config_dict,
    make_default_config
)
from smqtk_core.dict import merge_dict
from smqtk_dataprovider import DataElement
from smqtk_descriptors import DescriptorGenerator
from smqtk_descriptors.utils import parallel_map


LOG = logging.getLogger(__name__)


try:
    import PIL.Image  # type: ignore
    import PIL.ImageFile  # type: ignore
except ImportError as ex:
    LOG.warning(f"Failed to import PIL module: {ex}")
    PIL = None

try:
    import caffe  # type: ignore
except ImportError as ex:
    LOG.warning(f"Failed to import caffe module: {ex}")
    caffe = None


__all__ = [
    "CaffeDescriptorGenerator",
]
T = TypeVar("T", bound="CaffeDescriptorGenerator")


class CaffeDescriptorGenerator (DescriptorGenerator):
    """
    Compute images against a Caffe model, extracting a layer as the content
    descriptor.

    :param network_prototxt: Data element containing the text file defining
        the network layout.
    :param network_model: Data element containing the trained
        ``.caffemodel`` file to use.
    :param image_mean: Optional data element containing the image mean
        ``.binaryproto`` or ``.npy`` file.
    :param return_layer: The label of the layer we take data from to compose
        output descriptor vector.
    :param batch_size: The maximum number of images to process in one feed
        forward of the network. This is especially important for GPUs since
        they can only process a batch that will fit in the GPU memory space.
    :param use_gpu: If Caffe should try to use the GPU
    :param gpu_device_id: Integer ID of the GPU device to use. Only used if
        ``use_gpu`` is True.
    :param network_is_bgr: If the network is expecting BGR format pixels.
        For example, the BVLC default caffenet does (thus the default is
        True).
    :param data_layer: String label of the network's data layer.
        We assume its 'data' by default.
    :param load_truncated_images: If we should be lenient and force loading
        of truncated image bytes. This is False by default.
    :param pixel_rescale: Re-scale image pixel values before being
        transformed by caffe (before mean subtraction, etc)
        into the given tuple ``(min, max)`` range. By default, images are
        loaded in the ``[0, 255]`` range. Refer to the image mean being used
        for desired input pixel scale.
    :param input_scale: Optional floating-point scalar value to scale values
        of caffe network input data AFTER mean subtraction. This value is
        directly multiplied against the pixel values.
    :param threads:
        Optional specific number of threads to use for data loading and
        pre-processing. If this is None or 0, we introspect the current
        system thread capacity and use that.

    :raises AssertionError: Optionally provided image mean protobuf
        consisted of more than one image, or its shape was neither 1 nor 3
        channels.
    """

    @classmethod
    def is_usable(cls) -> bool:
        valid = caffe is not None and PIL is not None
        if not valid:
            LOG.debug("Caffe python module cannot be imported")
        return valid

    @classmethod
    def get_default_config(cls) -> Dict[str, Any]:
        default = super(CaffeDescriptorGenerator, cls).get_default_config()

        data_elem_impl_set = DataElement.get_impls()
        # Need to make copies of dict so changes to one does not effect others.
        default['network_prototxt'] = \
            make_default_config(data_elem_impl_set)
        default['network_model'] = make_default_config(data_elem_impl_set)
        default['image_mean'] = make_default_config(data_elem_impl_set)

        return default

    @classmethod
    def from_config(
        cls: Type[T],
        config_dict: Dict,
        merge_default: bool = True
    ) -> T:
        if merge_default:
            config_dict = merge_dict(cls.get_default_config(),
                                     config_dict)

        data_elem_impl_set = DataElement.get_impls()

        # Translate prototext and model sub-configs into DataElement instances.
        config_dict['network_prototxt'] = \
            from_config_dict(config_dict['network_prototxt'],
                             data_elem_impl_set)
        config_dict['network_model'] = \
            from_config_dict(config_dict['network_model'],
                             data_elem_impl_set)

        # Translate optionally provided image mean sub-config into a
        # DataElement instance. May have been provided as ``None`` or a
        # configuration dictionary with type ``None`.
        # None, dict[type=None], dict[type=str]
        if config_dict['image_mean'] is None \
                or config_dict['image_mean'].get('type', None) is None:
            config_dict['image_mean'] = None
        else:
            config_dict['image_mean'] = \
                from_config_dict(config_dict['image_mean'], data_elem_impl_set)

        return super(CaffeDescriptorGenerator, cls).from_config(config_dict, merge_default=False)

    def __init__(
        self,
        network_prototxt: DataElement,
        network_model: DataElement,
        image_mean: Optional[DataElement] = None,
        return_layer: str = 'fc7',
        batch_size: int = 1,
        use_gpu: bool = False,
        gpu_device_id: int = 0,
        network_is_bgr: bool = True,
        data_layer: str = 'data',
        load_truncated_images: bool = False,
        pixel_rescale: Optional[Tuple[float, float]] = None,
        input_scale: Optional[float] = None,
        threads: Optional[int] = None
    ):
        """
        Create a Caffe CNN descriptor generator
        """
        super(CaffeDescriptorGenerator, self).__init__()

        self.network_prototxt = network_prototxt
        self.network_model = network_model
        self.image_mean = image_mean

        self.return_layer = str(return_layer)
        self.batch_size = int(batch_size)

        self.use_gpu = bool(use_gpu)
        self.gpu_device_id = int(gpu_device_id)

        self.network_is_bgr = bool(network_is_bgr)
        self.data_layer = str(data_layer)

        self.load_truncated_images = bool(load_truncated_images)
        self.pixel_rescale = pixel_rescale
        self.input_scale = input_scale

        self.threads = threads

        assert self.batch_size > 0, \
            "Batch size must be greater than 0 (got %d)" \
            % self.batch_size
        assert self.gpu_device_id >= 0, \
            "GPU Device ID must be greater than 0 (got %d)" \
            % self. gpu_device_id

        # Network setup variables
        self.network: Optional[caffe.Net] = None
        self.net_data_shape = ()
        self.transformer: Optional[caffe.io.Transformer] = None

        self._setup_network()

    def __getstate__(self) -> Dict[str, Any]:
        return self.get_config()

    def __setstate__(self, state: Mapping[str, Any]) -> None:
        # This ``__dict__.update`` works because configuration parameters
        # exactly match up with instance attributes currently.
        self.__dict__.update(state)
        # Translate nested Configurable instance configurations into actual
        # object instances.
        self.network_prototxt = from_config_dict(
            state["network_prototxt"], DataElement.get_impls()
        )
        # noinspection PyTypeChecker
        self.network_model = from_config_dict(
            state["network_model"], DataElement.get_impls()
        )
        state_image_mean = state["image_mean"]
        if state_image_mean is not None:
            # noinspection PyTypeChecker
            self.image_mean = from_config_dict(
                state_image_mean, DataElement.get_impls()
            )
        self._setup_network()

    def _set_caffe_mode(self) -> None:
        """
        Set the appropriate Caffe mode on the current thread/process.
        """
        if self.use_gpu:
            LOG.debug("Using GPU")
            caffe.set_device(self.gpu_device_id)
            caffe.set_mode_gpu()
        else:
            LOG.debug("using CPU")
            caffe.set_mode_cpu()

    def _setup_network(self) -> None:
        """
        Initialize Caffe and the network

        :raises AssertionError: Optionally provided image mean protobuf
            consisted of more than one image, or its shape was neither 1 or 3
            channels.
        """
        self._set_caffe_mode()

        # Questions:
        #   - ``caffe.TEST`` indicates phase of either TRAIN or TEST
        LOG.debug("Initializing network")
        LOG.debug("Loading Caffe network from network/model configs")
        self.network = caffe.Net(
            self.network_prototxt.write_temp().encode(),
            caffe.TEST,
            weights=self.network_model.write_temp().encode()
        )
        self.network_prototxt.clean_temp()
        self.network_model.clean_temp()
        # Assuming the network has a 'data' layer and notion of data shape
        self.net_data_shape = self.network.blobs[self.data_layer].data.shape
        LOG.debug("Network data shape: %s", self.net_data_shape)

        # Crating input data transformer
        LOG.debug("Initializing data transformer")
        self.transformer = caffe.io.Transformer(
            {self.data_layer: self.network.blobs[self.data_layer].data.shape}
        )
        LOG.debug("Initializing data transformer -> %s",
                  self.transformer.inputs)

        if self.image_mean is not None:
            LOG.debug("Loading image mean (reducing to single pixel mean)")
            image_mean_bytes = self.image_mean.get_bytes()
            try:
                # noinspection PyTypeChecker
                a = numpy.load(BytesIO(image_mean_bytes), allow_pickle=True)
                LOG.info("Loaded image mean from numpy bytes")
            except IOError:
                LOG.debug("Image mean file not a numpy array, assuming URI to "
                          "protobuf binary.")
                # noinspection PyUnresolvedReferences
                blob = caffe.proto.caffe_pb2.BlobProto()
                blob.ParseFromString(image_mean_bytes)
                a = numpy.array(caffe.io.blobproto_to_array(blob))
                assert a.shape[0] == 1, \
                    "Input image mean blob protobuf consisted of more than " \
                    "one image. Not sure how to handle this yet."
                a = a.reshape(a.shape[1:])
                LOG.info("Loaded image mean from protobuf bytes")
            assert a.shape[0] in [1, 3], \
                "Currently asserting that we either get 1 or 3 channel " \
                "images. Got a %d channel image." % a[0]
            # TODO: Instead of always using pixel mean, try to use image-mean
            #       if given. Might have to rescale if image/data layer shape
            #       is different.
            a_mean = a.mean(1).mean(1)
            LOG.debug("Initializing data transformer -- mean")
            self.transformer.set_mean(self.data_layer, a_mean)

        LOG.debug("Initializing data transformer -- transpose")
        self.transformer.set_transpose(self.data_layer, (2, 0, 1))
        if self.network_is_bgr:
            LOG.debug("Initializing data transformer -- channel swap")
            self.transformer.set_channel_swap(self.data_layer, (2, 1, 0))
        if self.input_scale:
            LOG.debug("Initializing data transformer -- input scale")
            self.transformer.set_input_scale(self.data_layer, self.input_scale)

    def get_config(self) -> Dict[str, Any]:
        """
        Return a JSON-compliant dictionary that could be passed to this class's
        ``from_config`` method to produce an instance with identical
        configuration.

        In the common case, this involves naming the keys of the dictionary
        based on the initialization argument names as if it were to be passed
        to the constructor via dictionary expansion.

        :return: JSON type compliant configuration dictionary.
        """
        image_mean_config: Optional[Dict]
        if self.image_mean is not None:
            image_mean_config = to_config_dict(self.image_mean)
        else:
            image_mean_config = None
        return {
            "network_prototxt": to_config_dict(self.network_prototxt),
            "network_model": to_config_dict(self.network_model),
            "image_mean": image_mean_config,
            "return_layer": self.return_layer,
            "batch_size": self.batch_size,
            "use_gpu": self.use_gpu,
            "gpu_device_id": self.gpu_device_id,
            "network_is_bgr": self.network_is_bgr,
            "data_layer": self.data_layer,
            "load_truncated_images": self.load_truncated_images,
            "pixel_rescale": self.pixel_rescale,
            "input_scale": self.input_scale,
            "threads": self.threads,
        }

    def valid_content_types(self) -> Set[str]:
        """
        :return: A set valid MIME type content types that this descriptor can
            handle.
        """
        return {
            'image/tiff',
            'image/png',
            'image/jpeg',
            'image/bmp',
        }

    def _generate_arrays(self, data_iter: Iterable[DataElement]) -> Iterable[numpy.ndarray]:
        """
        Inner template method that defines the generation of descriptor vectors
        for a given iterable of data elements.

        Pre-conditions:
          - Data elements input to this method have been validated to be of at
            least one of this class's reported ``valid_content_types``.

        :param collections.abc.Iterable[DataElement] data_iter:
            Iterable of data element instances to be described.

        :raises RuntimeError: Descriptor extraction failure of some kind.

        :return: Iterable of numpy arrays in parallel association with the
            input data elements.
        :rtype: collections.abc.Iterable[numpy.ndarray]
        """
        assert self.network is not None, (
            "A network should be initialized by now."
        )

        self._set_caffe_mode()
        log_debug = LOG.debug

        # Start parallel operation to pre-process imagery before aggregating
        # for network execution.
        # TODO: update ``buffer_factor`` param to account for batch size?
        img_array_iter = \
            parallel_map(_process_load_img_array,
                         zip(
                             data_iter, itertools.repeat(self.transformer),
                             itertools.repeat(self.data_layer),
                             itertools.repeat(self.load_truncated_images),
                             itertools.repeat(self.pixel_rescale),
                         ),
                         ordered=True, cores=self.threads)

        # Aggregate and process batches of input data elements
        #: :type: list[numpy.ndarray]
        batch_img_arrays = \
            list(itertools.islice(img_array_iter, self.batch_size))
        batch_i = 0
        while len(batch_img_arrays) > 0:
            cur_batch_size = len(batch_img_arrays)
            log_debug("Batch {} - size {}".format(batch_i, cur_batch_size))

            log_debug("Updating network data layer shape ({} images)"
                      .format(cur_batch_size))
            self.network.blobs[self.data_layer].reshape(
                cur_batch_size, *self.net_data_shape[1:4]
            )
            log_debug("Loading image matrices into network layer '{:s}'"
                      .format(self.data_layer))
            self.network.blobs[self.data_layer].data[...] = batch_img_arrays
            log_debug("Moving network forward")
            self.network.forward()
            descriptor_list = self.network.blobs[self.return_layer].data
            log_debug("extracting return layer '{:s}' into vectors"
                      .format(self.return_layer))
            for v in descriptor_list:
                if v.ndim > 1:
                    # In case caffe generates multidimensional array
                    # like (rows, 1, 1)
                    log_debug("- Raveling output array of shape {}"
                              .format(v.shape))
                    yield numpy.ravel(v)
                else:
                    yield v

            # Slice out the next batch
            #: :type: list[(collections.abc.Hashable, numpy.ndarray)]
            batch_img_arrays = \
                list(itertools.islice(img_array_iter, self.batch_size))
            batch_i += 1


def _process_load_img_array(
    input_tuple: Tuple[DataElement, "caffe.io.Transformer", str, bool, Optional[Tuple[float, float]]]
) -> Tuple[Hashable, numpy.ndarray]:
    """
    Helper function for multiprocessing image data loading

    Expected input argument tuple contents (in tuple order):
        * data_element: DataElement providing bytes
        * transformer: Caffe Transformer instance for pre-processing.
        * data_layer: String label of the network's data layer
        * load_truncated_images: Boolean of whether loading truncated images is
          allowed (See PIL.ImageFile.LOAD_TRUNCATED_IMAGES attribute).
        * pixel_rescale: Pair of floating point values to recale image values
          into, i.e. [0, 255] (the default).

    :param input_tuple:
        Tuple of input arguments as we expect to be called by a multiprocessing
        map function. See above for content details.

    :return: Input DataElement UUID and Pre-processed numpy array.
    """
    # data_element: DataElement providing bytes
    # transformer: Caffe Transformer instance for pre-processing.
    # data_layer: String label of the data layer
    (data_element, transformer, data_layer, load_truncated_images,
     pixel_rescale) = input_tuple
    PIL.ImageFile.LOAD_TRUNCATED_IMAGES = load_truncated_images
    try:
        img = PIL.Image.open(BytesIO(data_element.get_bytes()))
    except Exception as ex_:
        logging.getLogger(__name__).error(
            "Failed opening image from data element {}. Exception ({}): {}"
            .format(data_element, type(ex_), str(ex_))
        )
        raise
    if img.mode != "RGB":
        img = img.convert("RGB")
    logging.getLogger(__name__).debug("Image: {}".format(img))
    # Caffe natively uses float types (32-bit)
    try:
        # This can fail if the image is truncated and we're not allowing the
        # loading of those images
        img_a = numpy.asarray(img, numpy.float32)
    except Exception as ex_:
        logging.getLogger(__name__).error(
            "Failed array-ifying data element {}. Image may be truncated. "
            "Exception ({}): {}"
            .format(data_element, type(ex_), str(ex_))
        )
        raise
    assert img_a.ndim == 3, \
        "Loaded invalid RGB image with shape {:s}".format(img_a.shape)
    if pixel_rescale:
        pmin, pmax = min(pixel_rescale), max(pixel_rescale)
        r = pmax - pmin
        img_a = (img_a / (255. / r)) + pmin
    img_at = transformer.preprocess(data_layer, img_a)
    return img_at
