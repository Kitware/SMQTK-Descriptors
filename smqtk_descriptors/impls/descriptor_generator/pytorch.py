import abc
import copy
import itertools
import logging
from typing import (
    Any, Callable, Dict, Iterable, Iterator, Literal, Mapping,
    Optional, Sequence, Set, Type, TypeVar, Union
)

import numpy as np

from smqtk_dataprovider import DataElement
from smqtk_descriptors import DescriptorGenerator
from smqtk_image_io import ImageReader
from smqtk_core.configuration import (
    from_config_dict,
    make_default_config,
    to_config_dict,
)
from smqtk_descriptors.utils.parallel import parallel_map
from smqtk_descriptors.utils.pytorch_utils import load_state_dict


LOG = logging.getLogger(__name__)

try:
    import torch  # type: ignore
    from torch.utils.data import Dataset  # type: ignore
    import torchvision.models  # type: ignore
    import torchvision.transforms  # type: ignore
    from torch.nn import functional as F  # type: ignore
except ModuleNotFoundError:
    torch = None  # type: ignore

__all__ = [
    "TorchModuleDescriptorGenerator",
    "Resnet50SequentialTorchDescriptorGenerator",
    "AlignedReIDResNet50TorchDescriptorGenerator"
]
T = TypeVar("T", bound="TorchModuleDescriptorGenerator")


def normalize_vectors(
    v: np.ndarray,
    mode: Optional[Union[int, float, Literal['fro', 'nuc']]] = None
) -> np.ndarray:
    """
    Array/Matrix normalization along max dimension (i.e. a=0 for 1D array, a=1
    for 2D array, etc.).

    :param v: Vector to normalize.
    :param mode: ``numpy.linalg.norm`` order parameter.

    :return: Normalize version of the input array ``v``.
    """
    if mode is not None:
        n = np.linalg.norm(v, mode, v.ndim - 1, keepdims=True)
        # replace 0's with 1's, preventing div-by-zero
        n[n == 0.] = 1.
        return v / n
    # When normalization off
    return v


try:
    class ImgMatDataset (Dataset):

        def __init__(
            self,
            img_mat_list: Sequence[np.ndarray],
            transform: Callable[[Iterable[np.ndarray]], "torch.Tensor"]
        ) -> None:
            self.img_mat_list = img_mat_list
            self.transform = transform

        def __getitem__(self, index: int) -> "torch.Tensor":
            return self.transform(self.img_mat_list[index])

        def __len__(self) -> int:
            return len(self.img_mat_list)
except NameError:
    pass


class TorchModuleDescriptorGenerator (DescriptorGenerator):
    """
    Descriptor generator using some torch module.

    :param image_reader:
        Image reader algorithm to use for image loading.
    :param image_load_threads:
        Number of threads to use for parallel image loading. Set to "None" to
        use all available CPU threads. This is 1 (serial) by default.
    :param weights_filepath:
        Optional filepath to saved network weights to load. If this value is
        None then implementations should attempt to use their "pre-trained"
        mode if they have one. Otherwise an error is raised on model load when
        not provided.
    :param image_tform_threads:
        Number of threads to utilize when transforming image matrices for
        network input. Set to `None` to use all available CPU threads.
        This is 1 (serial) by default).
    :param batch_size:
        Batch size use for model computation.
    :param use_gpu:
        If the model should be loaded onto, and processed on, the GPU.
    :param cuda_device:
        Specific device value to pass to the CUDA transfer calls if `use_gpu`
        us enabled.
    :param normalize:
        Optionally normalize descriptor vectors as they are produced. We use
        ``numpy.linalg.norm`` so any valid value for the ``ord`` parameter is
        acceptable here.
    :param iter_runtime:
        By default the input image matrix data is accumulated into a list in
        order to utilize `torch.utils.data.DataLoader` for batching.
        If this is parameter is True however, we use a custom parallel
        iteration pipeline for image transformation into network appropriate
        tensors.
        Using this mode changes the maximum RAM usage profile to be linear with
        batch-size instead of input image data amount, as well as having lower
        latency to first yield.
        This mode is idea for streaming input or high input volume situations.
        This mode currently has a short-coming of working only in a threaded
        manner so it is not as fast as using the `DataLoader` avenue.
    :param global_average_pool:
        Optionally apply a GAP operation to the spatial dimension of the feature
        vector that is returned by the descriptor generator. Some models return
        a w x h x k tensor and flatten them into a single dimension, which can
        remove the spatial context of the features. Taking an average over each
        channel can improve the clustering in some cases.
    """

    @classmethod
    def is_usable(cls) -> bool:
        valid = torch is not None
        if not valid:
            LOG.debug("Torch python module not imported")
        return valid

    @classmethod
    def get_default_config(cls) -> Dict[str, Any]:
        c = super().get_default_config()
        c['image_reader'] = make_default_config(ImageReader.get_impls())
        return c

    @classmethod
    def from_config(
        cls: Type[T],
        config_dict: Dict,
        merge_default: bool = True
    ) -> T:
        # Copy config to prevent input modification
        config_dict = copy.deepcopy(config_dict)
        config_dict['image_reader'] = from_config_dict(
            config_dict['image_reader'],
            ImageReader.get_impls()
        )
        return super().from_config(config_dict, merge_default)

    def __init__(
        self,
        image_reader: ImageReader,
        image_load_threads: Optional[int] = 1,
        weights_filepath: Optional[str] = None,
        image_tform_threads: Optional[int] = 1,
        batch_size: int = 32,
        use_gpu: bool = False,
        cuda_device: Optional[int] = None,
        normalize: Optional[Union[int, float, Literal['fro', 'nuc']]] = None,
        iter_runtime: bool = False,
        global_average_pool: bool = False
    ):
        super().__init__()

        self.image_reader = image_reader
        self.image_load_threads = image_load_threads
        self.weights_filepath = weights_filepath
        self.image_tform_threads = image_tform_threads
        self.batch_size = batch_size
        self.use_gpu = use_gpu
        self.cuda_device = cuda_device
        self.normalize = normalize
        self.iter_runtime = iter_runtime
        self.global_average_pool = global_average_pool
        # Place-holder for the torch.nn.Module loaded.
        self.module: Optional[torch.nn.Module] = None
        # Just load model on construction
        # - this may have issues in multi-threaded/processed contexts. Use
        #   will tell.
        self._ensure_module()

    @abc.abstractmethod
    def _load_module(self) -> "torch.nn.Module":
        """
        :return: Load and return the module.
        """

    @abc.abstractmethod
    def _make_transform(self) -> Callable[[Iterable[np.ndarray]], "torch.Tensor"]:
        """
        :returns: A callable that takes in a ``numpy.ndarray`` image matrix and
            returns a transformed version as a ``torch.Tensor``.
        """

    def __getstate__(self) -> Dict[str, Any]:
        return self.get_config()

    def __setstate__(self, state: Mapping[str, Any]) -> None:
        # This ``__dict__.update`` works because configuration parameters
        # exactly match up with instance attributes currently.
        self.__dict__.update(state)
        # Translate nested Configurable instance configurations into actual
        # object instances.
        self.image_reader = from_config_dict(
            state['image_reader'], ImageReader.get_impls()
        )
        self._ensure_module()

    def valid_content_types(self) -> Set:
        return self.image_reader.valid_content_types()

    def is_valid_element(self, data_element: DataElement) -> bool:
        # Check element validity though the ImageReader algorithm instance
        return self.image_reader.is_valid_element(data_element)

    def _ensure_module(self) -> "torch.nn.Module":
        if self.module is None:
            module = self._load_module()
            if self.weights_filepath:
                checkpoint = torch.load(self.weights_filepath,
                                        # In-case weights were saved with the
                                        # context of a non-CPU device.
                                        map_location="cpu")

                # A common alternative pattern is for training to save
                # checkpoints/state-dicts as a nested dict that contains the
                # state-dict under the key 'state_dict'.
                if 'state_dict' in checkpoint:
                    checkpoint = checkpoint['state_dict']
                elif 'state_dicts' in checkpoint:
                    checkpoint = checkpoint['state_dicts'][0]

                # Align model and checkpoint and load weights
                load_state_dict(module, checkpoint)

            module.eval()
            if self.use_gpu:
                module = module.cuda(self.cuda_device)
            self.module = module

        return self.module

    def _generate_arrays(self, data_iter: Iterable[DataElement]) -> Iterable[np.ndarray]:
        # Generically load image data [in parallel], iterating results into
        # template method.
        ir_load: Callable[..., Any] = self.image_reader.load_as_matrix
        i_load_threads = self.image_load_threads

        gen_fn = (
            self.generate_arrays_from_images_naive,     # False
            self.generate_arrays_from_images_iter,      # True
        )[self.iter_runtime]

        if i_load_threads is None or i_load_threads > 1:
            return gen_fn(
                parallel_map(ir_load, data_iter,
                             cores=i_load_threads)
            )
        else:
            return gen_fn(
                ir_load(d) for d in data_iter
            )

    # NOTE: may need to create wrapper function around _make_transform
    #       that adds a PIL.Image.from_array and convert transformation to
    #       ensure being in the expected image format for the network.

    def generate_arrays_from_images_naive(
        self,
        img_mat_iter: Iterable[np.ndarray]
    ) -> Iterable[np.ndarray]:
        model = self._ensure_module()

        # Just gather all input into list for pytorch dataset wrapping
        img_mat_list = list(img_mat_iter)
        tform_img_fn = self._make_transform()
        use_gpu = self.use_gpu
        cuda_device = self.cuda_device

        if self.image_tform_threads is not None:
            num_workers: int = min(self.image_tform_threads, len(img_mat_list))
        else:
            num_workers = len(img_mat_list)

        dl = torch.utils.data.DataLoader(
            ImgMatDataset(img_mat_list, tform_img_fn),
            batch_size=self.batch_size,
            # Don't need more workers than we have input, so don't bother
            # spinning them up...
            num_workers=num_workers,
            # Use pinned memory when we're in use-gpu mode.
            pin_memory=use_gpu is True)

        for batch_input in dl:
            # batch_input: batch_size x channels x height x width
            if use_gpu:
                batch_input = batch_input.cuda(cuda_device)

            feats = self._forward(model, batch_input)

            for f in feats:
                yield f

    def generate_arrays_from_images_iter(
        self,
        img_mat_iter: Iterable[DataElement]
    ) -> Iterable[np.ndarray]:
        """
        Template method for implementation to define descriptor generation over
        input image matrices.

        :param img_mat_iter:
            Iterable of numpy arrays representing input image matrices.

        :raises

        :return: Iterable of numpy arrays in parallel association with the
            input image matrices.
        """
        model = self._ensure_module()

        # Set up running parallelize
        # - Not utilizing a DataLoader due to input being an iterator where
        #   we don't know the size of input a priori.
        tform_img_fn: Callable[..., Any] = self._make_transform()
        tform_threads = self.image_tform_threads
        if tform_threads is None or tform_threads > 1:
            tfed_mat_iter: Iterator = parallel_map(tform_img_fn,
                                                   img_mat_iter,
                                                   cores=self.image_tform_threads,
                                                   name="tform_img",
                                                   ordered=True)
        else:
            tfed_mat_iter = (
                tform_img_fn(mat)
                for mat in img_mat_iter
            )

        batch_size = self.batch_size
        use_gpu = self.use_gpu
        # We don't know the shape of input data yet.
        batch_tensor = None

        #: :type: list[torch.Tensor]
        batch_slice = list(itertools.islice(tfed_mat_iter, batch_size))
        cuda_device = self.cuda_device
        while batch_slice:
            # Use batch tensor unless this is the leaf batch, in which case
            # we need to allocate a reduced size one.
            if batch_tensor is None or len(batch_slice) != batch_size:
                batch_size = len(batch_slice)
                batch_tensor = torch.empty([len(batch_slice)] +
                                           list(batch_slice[0].shape))
                if use_gpu:
                    # When using the GPU, attempt to use page-locked memory.
                    # https://devblogs.nvidia.com/how-optimize-data-transfers-cuda-cc/
                    batch_tensor.pin_memory()

            # Fill in tensor with batch image matrices
            for i in range(batch_size):
                batch_tensor[i, :] = batch_slice[i]
            process_tensor = batch_tensor
            if use_gpu:
                # Copy input data tensor in batch to GPU.
                # - Need to use the same CUDA device that model is loaded on.
                process_tensor = batch_tensor.cuda(cuda_device)

            feats = self._forward(model, process_tensor)

            for f in feats:
                yield f
            #: :type: list[torch.Tensor]
            batch_slice = list(itertools.islice(tfed_mat_iter, batch_size))

    def _forward(self, model: "torch.nn.Module", model_input: "torch.Tensor") -> np.ndarray:
        """
        Template method for implementation of forward pass of model.

        :param model: Network module with loaded weights
        :param model_input: Tensor that has been appropriately
            shaped and placed onto target inference hardware.

        :return: Tensor output of module() call
        """
        with torch.no_grad():
            feats = model(model_input)

        # Apply global average pool
        if self.global_average_pool and len(feats.size()) > 2:
            feats = F.avg_pool2d(feats, feats.size()[2:])
            feats = feats.view(feats.size(0), -1)

        feats_np = np.squeeze(feats.cpu().numpy().astype(np.float32))
        if len(feats_np.shape) < 2:
            # Add a dim if the batch size was only one (first dim squeezed
            # down).
            feats_np = np.expand_dims(feats_np, 0)

        # Normalizing *after* squeezing for axis sanity.
        feats_np = normalize_vectors(feats_np, self.normalize)

        return feats_np

    def get_config(self) -> Dict[str, Any]:
        return {
            "image_reader": to_config_dict(self.image_reader),
            "image_load_threads": self.image_load_threads,
            "weights_filepath": self.weights_filepath,
            "image_tform_threads": self.image_tform_threads,
            "batch_size": self.batch_size,
            "use_gpu": self.use_gpu,
            "cuda_device": self.cuda_device,
            "normalize": self.normalize,
            "iter_runtime": self.iter_runtime,
            "global_average_pool": self.global_average_pool
        }


class Resnet50SequentialTorchDescriptorGenerator (TorchModuleDescriptorGenerator):
    """
    Use torchvision.models.resnet50, but chop off the final fully-connected
    layer as a ``torch.nn.Sequential``.
    """

    @classmethod
    def is_usable(cls) -> bool:
        valid = torch is not None
        if not valid:
            LOG.debug("Torch python module not imported")
        return valid

    def _load_module(self) -> "torch.nn.Module":
        pretrained = self.weights_filepath is None
        m = torchvision.models.resnet50(
            pretrained=pretrained
        )
        if pretrained:
            LOG.info("Using pre-trained weights for pytorch ResNet-50 "
                     "network.")
        return torch.nn.Sequential(
            *tuple(m.children())[:-1]
        )

    def _make_transform(self) -> Callable[[Iterable[np.ndarray]], "torch.Tensor"]:
        # Transform based on: https://pytorch.org/hub/pytorch_vision_resnet/
        return torchvision.transforms.Compose([
            torchvision.transforms.ToPILImage(mode='RGB'),
            torchvision.transforms.Resize((224, 224)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])


class AlignedReIDResNet50TorchDescriptorGenerator (TorchModuleDescriptorGenerator):
    """
    Descriptor generator for computing Aligned Re-ID++ descriptors.
    """

    @classmethod
    def is_usable(cls) -> bool:
        valid = torch is not None
        if not valid:
            LOG.debug("Torch python module not imported")
        return valid

    def _load_module(self) -> "torch.nn.Module":
        pretrained = self.weights_filepath is None
        # Pre-initialization with imagenet model important - provided
        # checkpoint potentially missing layers
        m = torchvision.models.resnet50(
            pretrained=True
        )
        if pretrained:
            LOG.info("Using pre-trained weights for pytorch ResNet-50 "
                     "network.")

        # Return model without pool and linear layer
        return torch.nn.Sequential(
            *tuple(m.children())[:-2]
        )

    def _forward(self, model: "torch.nn.Module", model_input: "torch.Tensor") -> np.ndarray:
        with torch.no_grad():
            feats = model(model_input)

        # Use only global features from return of (global_feats, local_feats)
        if isinstance(feats, tuple):
            feats = feats[0]

        feats = F.avg_pool2d(feats, feats.size()[2:])
        feats = feats.view(feats.size(0), -1)

        feats_np = np.squeeze(feats.cpu().numpy().astype(np.float32))
        if len(feats_np.shape) < 2:
            # Add a dim if the batch size was only one (first dim squeezed
            # down).
            feats_np = np.expand_dims(feats_np, 0)

        # Normalizing *after* squeezing for axis sanity.
        feats_np = normalize_vectors(feats_np, self.normalize)

        return feats_np

    def _make_transform(self) -> Callable[[Iterable[np.ndarray]], "torch.Tensor"]:
        # Transform based on: https://pytorch.org/hub/pytorch_vision_resnet/
        return torchvision.transforms.Compose([
            torchvision.transforms.ToPILImage(mode='RGB'),
            torchvision.transforms.Resize((224, 224)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
