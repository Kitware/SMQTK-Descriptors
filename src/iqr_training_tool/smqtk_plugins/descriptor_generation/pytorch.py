import abc
import copy
import itertools

import numpy as np

from smqtk.algorithms import DescriptorGenerator, ImageReader
from smqtk.utils.configuration import (
    from_config_dict,
    make_default_config,
    to_config_dict,
)

from .utils import load_state_dict
from smqtk.utils.parallel import parallel_map

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.models
import torchvision.transforms
from torch.nn import functional as F


def normalize_vectors(v, mode=None):
    """
    Array/Matrix normalization along max dimension (i.e. a=0 for 1D array, a=1
    for 2D array, etc.).

    :param np.ndarray v:
        Vector to normalize.
    :param None|int|float|str mode:
        ``numpy.linalg.norm`` order parameter.

    :return: Normalize version of the input array ``v``.
    :rtype: np.ndarray
    """
    if mode is not None:
        n = np.linalg.norm(v, mode, v.ndim - 1, keepdims=True)
        # replace 0's with 1's, preventing div-by-zero
        n[n == 0.] = 1.
        return v / n
    # When normalization off
    return v

class ImgMatDataset (Dataset):

    def __init__(self, img_mat_list, transform):
        self._img_mat_list = img_mat_list
        self._transform = transform

    def __getitem__(self, index):
        return self._transform(self._img_mat_list[index])

    def __len__(self):
        return len(self._img_mat_list)


class TorchModuleDescriptorGenerator (DescriptorGenerator):
    """
    Descriptor generator using some torch module.

    :param smqtk.algorithms.ImageReader image_reader:
        Image reader algorithm to use for image loading.
    :param int|None image_load_threads:
        Number of threads to use for parallel image loading. Set to "None" to
        use all available CPU threads. This is 1 (serial) by default.
    :param None|str weights_filepath:
        Optional filepath to saved network weights to load. If this value is
        None then implementations should attempt to use their "pre-trained"
        mode if they have one. Otherwise an error is raised on model load when
        not provided.
    :param int|None image_tform_threads:
        Number of threads to utilize when transforming image matrices for
        network input. Set to `None` to use all available CPU threads.
        This is 1 (serial) by default).
    :param int batch_size:
        Batch size use for model computation.
    :param bool use_gpu:
        If the model should be loaded onto, and processed on, the GPU.
    :param None|int cuda_device:
        Specific device value to pass to the CUDA transfer calls if `use_gpu`
        us enabled.
    :param None|int|float|str normalize:
        Optionally normalize descriptor vectors as they are produced. We use
        ``numpy.linalg.norm`` so any valid value for the ``ord`` parameter is
        acceptable here.
    :param bool iter_runtime:
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
    :param bool global_average_pool:
        Optionally apply a GAP operation to the spatial dimension of the feature
        vector that is returned by the descriptor generator. Some models return
        a w x h x k tensor and flatten them into a single dimension, which can
        remove the spatial context of the features. Taking an average over each
        channel can improve the clustering in some cases.
    """

    def __init__(self, image_reader, image_load_threads=1,
                 weights_filepath=None, image_tform_threads=1, batch_size=32,
                 use_gpu=False, cuda_device=None, normalize=None,
                 iter_runtime=False, global_average_pool=False):
        super().__init__()
        self._image_reader = image_reader
        self._image_load_threads = image_load_threads
        self._weights_filepath = weights_filepath
        self._image_tform_threads = image_tform_threads
        self._batch_size = batch_size
        self._use_gpu = use_gpu
        self._cuda_device = cuda_device
        self._normalize = normalize
        self._iter_runtime = iter_runtime
        self._global_average_pool = global_average_pool
        # Place-holder for the torch.nn.Module loaded.
        self._module = None
        # Just load model on construction
        # - this may have issues in multi-threaded/processed contexts. Use will
        #   will tell.
        self._ensure_module()

    @abc.abstractmethod
    def _load_module(self):
        """
        :return: Load and return the module.
        :rtype: torch.nn.Module
        """

    @abc.abstractmethod
    def _make_transform(self):
        """
        :returns: A callable that takes in a ``numpy.ndarray`` image matrix and
            returns a transformed version as a ``torch.Tensor``.
        :rtype: (numpy.ndarray) -> torch.Tensor
        """

    def valid_content_types(self):
        return self._image_reader.valid_content_types()

    def is_valid_element(self, data_element):
        # Check element validity though the ImageReader algorithm instance
        return self._image_reader.is_valid_element(data_element)

    def _ensure_module(self):
        if self._module is None:
            module = self._load_module()
            if self._weights_filepath:
                checkpoint = torch.load(self._weights_filepath,
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
                #module.load_state_dict(checkpoint)

                # Align model and checkpoint and load weights
                load_state_dict(module, checkpoint) 

            module.eval()
            if self._use_gpu:
                module = module.cuda(self._cuda_device)
            self._module = module

        return self._module

    def _generate_arrays(self, data_iter):
        # Generically load image data [in parallel], iterating results into
        # template method.
        ir_load = self._image_reader.load_as_matrix
        i_load_threads = self._image_load_threads

        gen_fn = (
            self.generate_arrays_from_images_naive,     # False
            self.generate_arrays_from_images_iter,      # True
        )[self._iter_runtime]

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

    def generate_arrays_from_images_naive(self, img_mat_iter):
        model = self._ensure_module()

        # Just gather all input into list for pytorch dataset wrapping
        img_mat_list = list(img_mat_iter)
        tform_img_fn = self._make_transform()
        use_gpu = self._use_gpu
        cuda_device = self._cuda_device

        dl = torch.utils.data.DataLoader(
            ImgMatDataset(img_mat_list, tform_img_fn),
            batch_size=self._batch_size,
            # Don't need more workers than we have input, so don't bother
            # spinning them up...
            num_workers=min(self._image_tform_threads, len(img_mat_list)),
            # Use pinned memory when we're in use-gpu mode.
            pin_memory=use_gpu is True)
        with torch.no_grad():
            for batch_input in dl:
                # batch_input: batch_size x channels x height x width
                if use_gpu:
                    batch_input = batch_input.cuda(cuda_device)
                feats = model(batch_input)

                feats_np = np.squeeze(feats.cpu().numpy().astype(np.float32))
                if len(feats_np.shape) < 2:
                    # Add a dim if the batch size was only one (first dim
                    # squeezed down).
                    feats_np = np.expand_dims(feats_np, 0)
                # feats_np at this point: batch_size x n_feats
                # Normalizing *after* squeezing for axis sanity.
                feats_np = normalize_vectors(feats_np, self._normalize)
                for f in feats_np:
                    yield f

    def generate_arrays_from_images_iter(self, img_mat_iter):
        """
        Template method for implementation to define descriptor generation over
        input image matrices.

        :param collections.Iterable[numpy.ndarray] img_mat_iter:
            Iterable of numpy arrays representing input image matrices.

        :raises RuntimeError: Descriptor extraction failure of some kind.

        :return: Iterable of numpy arrays in parallel association with the
            input image matrices.
        :rtype: collections.Iterable[numpy.ndarray]
        """
        model = self._ensure_module()

        # Set up running parallelize
        # - Not utilizing a DataLoader due to input being an iterator where
        #   we don't know the size of input a priori.
        tform_img_fn = self._make_transform()
        tform_threads = self._image_tform_threads
        if tform_threads is None or tform_threads > 1:
            tfed_mat_iter = parallel_map(tform_img_fn,
                                         img_mat_iter,
                                         cores=self._image_tform_threads,
                                         name="{}_tform_img".format(self.name),
                                         ordered=True)
        else:
            tfed_mat_iter = (
                tform_img_fn(mat)
                for mat in img_mat_iter
            )

        batch_size = self._batch_size
        use_gpu = self._use_gpu
        # We don't know the shape of input data yet.
        batch_tensor = None

        #: :type: list[torch.Tensor]
        batch_slice = list(itertools.islice(tfed_mat_iter, batch_size))
        cuda_device = self._cuda_device
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
            with torch.no_grad():
                feats = model(process_tensor)

            # Check for models that return features for both (global, local)
            if isinstance(feats, tuple):
                feats = feats[0]

            # Apply global average pool
            if self._global_average_pool:
                feats = F.avg_pool2d(feats, feats.size()[2:])
                feats = feats.view(feats.size(0), -1)

            feats_np = np.squeeze(feats.cpu().numpy().astype(np.float32))
            if len(feats_np.shape) < 2:
                # Add a dim if the batch size was only one (first dim squeezed
                # down).
                feats_np = np.expand_dims(feats_np, 0)

            # Normalizing *after* squeezing for axis sanity.
            feats_np = normalize_vectors(feats_np, self._normalize)
            for f in feats_np:
                yield f
            #: :type: list[torch.Tensor]
            batch_slice = list(itertools.islice(tfed_mat_iter, batch_size))

    # Configuration overrides
    @classmethod
    def get_default_config(cls):
        c = super().get_default_config()
        c['image_reader'] = make_default_config(ImageReader.get_impls())
        return c

    @classmethod
    def from_config(cls, config_dict, merge_default=True):
        # Copy config to prevent input modification
        config_dict = copy.deepcopy(config_dict)
        config_dict['image_reader'] = \
            from_config_dict(config_dict['image_reader'],
                             ImageReader.get_impls())
        return super().from_config(config_dict, merge_default)

    def get_config(self):
        return {
            "image_reader": to_config_dict(self._image_reader),
            "image_load_threads": self._image_load_threads,
            "weights_filepath": self._weights_filepath,
            "image_tform_threads": self._image_tform_threads,
            "batch_size": self._batch_size,
            "use_gpu": self._use_gpu,
            "cuda_device": self._cuda_device,
        }

class Resnet50SequentualTorchDescriptorGenerator (TorchModuleDescriptorGenerator):
    """
    Use torchvision.models.resnet50, but chop off the final fully-connected
    layer as a ``torch.nn.Sequential``.
    """

    @classmethod
    def is_usable(cls):
        return True

    def _load_module(self):
        pretrained = self._weights_filepath is None
        m = torchvision.models.resnet50(
            pretrained=pretrained
        )
        if pretrained:
            self._log.info("Using pre-trained weights for pytorch ResNet-50 "
                           "network.".format(type(m)))
        return torch.nn.Sequential(
            *tuple(m.children())[:-1]
        )

    def _make_transform(self):
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
    def is_usable(cls):
        return True

    def _load_module(self):

        # Pre-initialization with imagenet model important - provided 
        # checkpoint potentially missing layers
        m = torchvision.models.resnet50(
            pretrained=True
        )
        if pretrained:
            self._log.info("Using pre-trained weights for pytorch ResNet-50 "
                           "network.".format(type(m)))

        # Return model without pool and linear layer
        return torch.nn.Sequential(
            *tuple(m.children())[:-2]
        )

    def _make_transform(self):
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

