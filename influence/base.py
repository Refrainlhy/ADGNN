import abc
from typing import Any, List, Optional

import numpy as np
import torch
from torch import nn
from torch.utils import data


def _set_attr(obj, names, val):
    if len(names) == 1:
        setattr(obj, names[0], val)
    else:
        _set_attr(getattr(obj, names[0]), names[1:], val)


def _del_attr(obj, names):
    if len(names) == 1:
        delattr(obj, names[0])
    else:
        _del_attr(getattr(obj, names[0]), names[1:])


class BaseObjective(abc.ABC):

    @abc.abstractmethod
    def train_outputs(self, model: nn.Module, batch: Any) -> torch.Tensor:

        raise NotImplementedError()

    @abc.abstractmethod
    def train_loss_on_outputs(self, outputs: torch.Tensor, batch: Any) -> torch.Tensor:

        raise NotImplementedError()

    @abc.abstractmethod
    def train_regularization(self, params: torch.Tensor) -> torch.Tensor:

        raise NotImplementedError()

    def train_loss(self, model: nn.Module, params: torch.Tensor, batch: Any) -> torch.Tensor:

        outputs = self.train_outputs(model, batch)
        return self.train_loss_on_outputs(outputs, batch) + self.train_regularization(params)

    @abc.abstractmethod
    def test_loss(self, model: nn.Module, params: torch.Tensor, batch: Any) -> torch.Tensor:

        raise NotImplementedError()


class BaseInfluenceModule(abc.ABC):
    def __init__(
            self,
            model: nn.Module,
            objective: BaseObjective,
            train_loader: data.DataLoader,
            test_loader: data.DataLoader,
            device: torch.device
    ):
        model.eval()
        self.model = model.to(device)
        self.device = device

        self.is_model_functional = False
        self.params_names = tuple(name for name, _ in self._model_params())
        self.params_shape = tuple(p.shape for _, p in self._model_params())

        self.objective = objective
        self.train_loader = train_loader
        self.test_loader = test_loader

    @abc.abstractmethod
    def inverse_hvp(self, vec: torch.Tensor) -> torch.Tensor:

        raise NotImplementedError()

    def train_loss_grad(self, train_idxs: List[int]) -> torch.Tensor:
      
        return self._loss_grad(train_idxs, train=True)

    def test_loss_grad(self, test_idxs: List[int]) -> torch.Tensor:
        
        return self._loss_grad(test_idxs, train=False)

    def stest(self, test_idxs: List[int]) -> torch.Tensor:
       

        return self.inverse_hvp(self.test_loss_grad(test_idxs))

    def influences(
            self,
            train_idxs: List[int],
            test_idxs: List[int],
            stest: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
  

        stest = self.stest(test_idxs) if (stest is None) else stest.to(self.device)

        scores = []
        for grad_z, _ in self._loss_grad_loader_wrapper(batch_size=1, subset=train_idxs, train=True):
            s = grad_z @ stest
            scores.append(s)
        return torch.tensor(scores) / len(self.train_loader.dataset)



    def _model_params(self, with_names=True):
        assert not self.is_model_functional
        return tuple((name, p) if with_names else p for name, p in self.model.named_parameters() if p.requires_grad)

    def _model_make_functional(self):
        assert not self.is_model_functional
        params = tuple(p.detach().requires_grad_() for p in self._model_params(False))

        for name in self.params_names:
            _del_attr(self.model, name.split("."))
        self.is_model_functional = True

        return params

    def _model_reinsert_params(self, params, register=False):
        for name, p in zip(self.params_names, params):
            _set_attr(self.model, name.split("."), torch.nn.Parameter(p) if register else p)
        self.is_model_functional = not register

    def _flatten_params_like(self, params_like):
        vec = []
        for p in params_like:
            vec.append(p.view(-1))
        return torch.cat(vec)

    def _reshape_like_params(self, vec):
        pointer = 0
        split_tensors = []
        for dim in self.params_shape:
            num_param = dim.numel()
            split_tensors.append(vec[pointer: pointer + num_param].view(dim))
            pointer += num_param
        return tuple(split_tensors)

    # Data helpers

    def _transfer_to_device(self, batch):
        if isinstance(batch, torch.Tensor):
            return batch.to(self.device)
        elif isinstance(batch, (tuple, list)):
            return type(batch)(self._transfer_to_device(x) for x in batch)
        elif isinstance(batch, dict):
            return {k: self._transfer_to_device(x) for k, x in batch.items()}
        else:
            raise NotImplementedError()

    def _loader_wrapper(self, train, batch_size=None, subset=None, sample_n_batches=-1):
        loader = self.train_loader if train else self.test_loader
        batch_size = loader.batch_size if (batch_size is None) else batch_size

        if subset is None:
            dataset = loader.dataset
        else:
            subset = np.array(subset)
            if len(subset.shape) != 1 or len(np.unique(subset)) != len(subset):
                raise ValueError()
            if np.any((subset < 0) | (subset >= len(loader.dataset))):
                raise IndexError()
            dataset = data.Subset(loader.dataset, indices=subset)

        if sample_n_batches > 0:
            num_samples = sample_n_batches * batch_size
            sampler = data.RandomSampler(data_source=dataset, replacement=True, num_samples=num_samples)
        else:
            sampler = None

        new_loader = data.DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=False,
            sampler=sampler,
            collate_fn=loader.collate_fn,
            num_workers=loader.num_workers,
            worker_init_fn=loader.worker_init_fn,
        )

        data_left = len(dataset)
        for batch in new_loader:
            batch = self._transfer_to_device(batch)
            size = min(batch_size, data_left)  # deduce batch size
            yield batch, size
            data_left -= size

    # Loss and autograd helpers

    def _loss_grad_loader_wrapper(self, train, **kwargs):
        params = self._model_params(with_names=False)
        flat_params = self._flatten_params_like(params)

        for batch, batch_size in self._loader_wrapper(train=train, **kwargs):
            loss_fn = self.objective.train_loss if train else self.objective.test_loss
            loss = loss_fn(model=self.model, params=flat_params, batch=batch)
            yield self._flatten_params_like(torch.autograd.grad(loss, params)), batch_size

    def _loss_grad(self, idxs, train):
        grad = 0.0
        for grad_batch, batch_size in self._loss_grad_loader_wrapper(subset=idxs, train=train):
            grad = grad + grad_batch * batch_size
        return grad / len(idxs)

    def _hvp_at_batch(self, batch, flat_params, vec, gnh):

        def f(theta_):
            self._model_reinsert_params(self._reshape_like_params(theta_))
            return self.objective.train_loss(self.model, theta_, batch)

        def out_f(theta_):
            self._model_reinsert_params(self._reshape_like_params(theta_))
            return self.objective.train_outputs(self.model, batch)

        def loss_f(out_):
            return self.objective.train_loss_on_outputs(out_, batch)

        def reg_f(theta_):
            return self.objective.train_regularization(theta_)

        if gnh:
            y, jvp = torch.autograd.functional.jvp(out_f, flat_params, v=vec)
            hjvp = torch.autograd.functional.hvp(loss_f, y, v=jvp)[1]
            gnhvp_batch = torch.autograd.functional.vjp(out_f, flat_params, v=hjvp)[1]
            return gnhvp_batch + torch.autograd.functional.hvp(reg_f, flat_params, v=vec)[1]
        else:
            return torch.autograd.functional.hvp(f, flat_params, v=vec)[1]
