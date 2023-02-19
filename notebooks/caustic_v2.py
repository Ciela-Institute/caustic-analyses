from abc import abstractmethod
from collections import OrderedDict
from math import pi, prod
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from caustic.constants import G_over_c2, arcsec_to_rad, c_Mpc_s, km_to_mpc
from caustic.cosmology import (
    Om0_default,
    _comoving_dist_helper_x_grid,
    _comoving_dist_helper_y_grid,
    h0_default,
    rho_cr_0_default,
)
from caustic.utils import derotate, to_elliptical, translate_rotate
from torch import Tensor
from torchinterp1d import interp1d


# Basic building blocks
class Param:
    """
    Represents a static or dynamic parameter. Static parameters are fixed while
    dynamic parameters must be passed in each time they're required.
    """

    def __init__(
        self,
        value: Optional[Tensor] = None,
        shape: Tuple[int, ...] = (),
    ):
        # Must assign one of value or shape
        self._value = value
        if value is None:
            self._shape = shape
        else:
            if shape is not None and shape != value.shape:
                raise ValueError(
                    f"value's shape {value.shape} does not match provided shape {shape}"
                )

            self._value = value
            self._shape = shape

    @property
    def dynamic(self):
        return self._value is None

    @property
    def value(self):
        return self._value

    @property
    def shape(self):
        return self._shape

    def __repr__(self):
        if not self.dynamic:
            return f"Param(value={self.value})"
        else:
            return f"Param(shape={self.shape})"


class Parametrized:
    """
    Represents a class with Param and Parametrized attributes.
    """
    def __init__(self, name: str):
        self.name = name
        self._params = OrderedDict()
        self._dynamic_size = 0
        self._n_dynamic = 0
        self._n_static = 0

        self._children = OrderedDict()

    def add_param(self, name, value: Optional[Tensor], shape: Tuple[int, ...] = ()):
        """
        Stores parameter in _params and records its size.
        """
        self._params[name] = Param(value, shape)
        if value is None:
            size = prod(shape)
            self._dynamic_size += size
            self._n_dynamic += 1
        else:
            self._n_static += 1

    def __setattr__(self, key, val):
        """
        When val is Parametrized, stores it in _children and adds it as an attribute.
        """
        if isinstance(val, Param):
            raise ValueError("cannot add Params directly as attributes: use add_param instead")

        if isinstance(val, Parametrized):
            self._children[val.name] = val
            for name, parametrized in val._children.items():
                self._children[name] = parametrized

        super().__setattr__(key, val)

    @property
    def n_dynamic(self) -> int:
        """
        Number of dynamic arguments in this object.
        """
        return self._n_dynamic

    @property
    def n_static(self) -> int:
        """
        Number of static arguments in this object.
        """
        return self._n_static

    @property
    def dynamic_size(self) -> int:
        """
        Total number of dynamic values in this object.
        """
        return self._dynamic_size

    def x_to_dict(
        self, x: Union[Dict[str, Union[List, Tensor, Dict[str, Tensor]]], List, Tensor]
    ) -> Dict[str, Any]:
        """
        Converts a list or tensor into a dict that can subsequently be unpacked
        into arguments to this object and its children.
        """
        # TODO: could probably give names of missing arguments...
        if isinstance(x, Dict):
            # Assume we don't need to repack
            return x
        elif isinstance(x, List):
            n_passed = len(x)
            n_expected = (
                sum([child.n_dynamic for child in self._children.values()])
                + self.n_dynamic
            )
            if n_passed != n_expected:
                raise ValueError(
                    f"{n_passed} dynamic args were passed, but {n_expected} are "
                    "required"
                )

            cur_offset = self.n_dynamic
            x_repacked = {self.name: x[:cur_offset]}
            for name, child in self._children.items():
                x_repacked[name] = x[cur_offset : cur_offset + child.n_dynamic]
                cur_offset += child.n_dynamic

            return x_repacked
        elif isinstance(x, Tensor):
            n_passed = x.shape[-1]
            n_expected = (
                sum([child.dynamic_size for child in self._children.values()])
                + self.dynamic_size
            )
            if n_passed != n_expected:
                raise ValueError(
                    f"{n_passed} flattened dynamic args were passed, but {n_expected}"
                    " are required"
                )

            cur_offset = self.dynamic_size
            x_repacked = {self.name: x[..., :cur_offset]}
            for name, child in self._children.items():
                x_repacked[name] = x[..., cur_offset : cur_offset + child.dynamic_size]
                cur_offset += child.dynamic_size

            return x_repacked
        else:
            raise ValueError("can only repack a list or 1D tensor")

    def unpack(self, x: Dict[str, Union[List, Tensor, Dict[str, Tensor]]]) -> List:
        """
        Unpacks a dict of kwargs, list of args or flattened vector of args to retrieve
        this object's static and dynamic parameters.
        """
        my_x = x[self.name]
        if isinstance(my_x, Dict):
            # TODO: validate to avoid collisons with static params
            return [
                value if value is not None else my_x[name]
                for name, value in self._params.items()
            ]
        elif isinstance(my_x, List):
            vals = []
            i = 0
            for param in self._params.values():
                if not param.dynamic:
                    vals.append(param.value)
                else:
                    vals.append(my_x[i])
                    i += 1

            return vals
        elif isinstance(my_x, Tensor):
            vals = []
            i = 0
            for param in self._params.values():
                if not param.dynamic:
                    vals.append(param.value)
                else:
                    size = prod(param.shape)
                    vals.append(my_x[..., i : i + size].reshape(param.shape))
                    i += size

            return vals
        else:
            raise ValueError(
                f"invalid argument type: must be a dict containing key {self.name} "
                "and value containing args as list or flattened tensor, or kwargs"
            )

    def __getattribute__(self, key):
        """
        Enables accessing static params as attributes.
        """
        try:
            return super().__getattribute__(key)
        except AttributeError as e:
            try:
                param = self._params[key]
                if not param.dynamic:
                    return param
            except:
                raise e

    @property
    def params(self) -> Tuple[OrderedDict, OrderedDict]:
        """
        Gets all of the static and dynamic Params for this object *and its children*.
        """
        static = OrderedDict()
        dynamic = OrderedDict()
        for name, param in self._params.items():
            if param.dynamic:
                dynamic[name] = param
            else:
                static[name] = param

        # Include childrens' parameters
        for name, child in self._children.items():
            child_static, child_dynamic = child.params
            static.update(child_static)
            dynamic.update(child_dynamic)

        return static, dynamic

    @property
    def static_params(self):
        return self.params[0]

    @property
    def dynamic_params(self):
        return self.params[1]

    def __str__(self):
        """
        String representation listing this object's name and the parameters for
        it and its children in the order they must be packed.
        """
        static, dynamic = self.params
        static = list(static.keys())
        dynamic = list(dynamic.keys())
        return (
            f"{self.__class__.__name__}(\n"
            f"    name='{self.name}',\n"
            f"    static params={static},\n"
            f"    dynamic params={dynamic}"
            "\n)"
        )


# Sources
class Sersic(Parametrized):
    def __init__(
        self,
        name: str,
        thx0=None,
        thy0=None,
        q=None,
        phi=None,
        index=None,
        th_e=None,
        I_e=None,
        s=torch.tensor(0.0),
        use_lenstronomy_k=False,
    ):
        super().__init__(name)

        # Shape defaults to scalar
        self.add_param("thx0", thx0)
        self.add_param("thy0", thy0)
        self.add_param("q", q)
        self.add_param("phi", phi)
        self.add_param("index", index)
        self.add_param("th_e", th_e)
        self.add_param("I_e", I_e)
        self.add_param("s", s)

        self.lenstronomy_k_mode = use_lenstronomy_k

    def brightness(self, thx, thy, x=[]):
        thx0, thy0, q, phi, index, th_e, I_e, s = self.unpack(x)

        thx, thy = translate_rotate(thx, thy, thx0, thy0, phi)
        ex, ey = to_elliptical(thx, thy, q)
        e = (ex**2 + ey**2).sqrt() + s

        if self.lenstronomy_k_mode:
            k = 1.9992 * index - 0.3271
        else:
            k = 2 * index - 1 / 3 + 4 / 405 / index + 46 / 25515 / index**2

        exponent = -k * ((e / th_e) ** (1 / index) - 1)
        return I_e * exponent.exp()


# Cosmology
class Cosmology(Parametrized):
    def __init__(self, name: str):
        super().__init__(name)

    @abstractmethod
    def rho_cr(self, z, x) -> torch.Tensor:
        ...

    @abstractmethod
    def comoving_dist(self, z, x) -> torch.Tensor:
        ...

    def comoving_dist_z1z2(self, z1, z2, x=[]):
        return self.comoving_dist(z2, x) - self.comoving_dist(z1, x)

    def angular_diameter_dist(self, z, x=[]):
        return self.comoving_dist(z, x) / (1 + z)

    def angular_diameter_dist_z1z2(self, z1, z2, x=[]):
        return self.comoving_dist_z1z2(z1, z2, x) / (1 + z2)

    def time_delay_dist(self, z_l, z_s, x=[]):
        d_l = self.angular_diameter_dist(z_l, x)
        d_s = self.angular_diameter_dist(z_s, x)
        d_ls = self.angular_diameter_dist_z1z2(z_l, z_s, x)
        return (1 + z_l) * d_l * d_s / d_ls

    def Sigma_cr(self, z_l, z_s, x=[]):
        d_l = self.angular_diameter_dist(z_l, x)
        d_s = self.angular_diameter_dist(z_s, x)
        d_ls = self.angular_diameter_dist_z1z2(z_l, z_s, x)
        return d_s / d_l / d_ls / (4 * pi * G_over_c2)


class FlatLambdaCDMCosmology(Cosmology):
    def __init__(
        self,
        name: str,
        h0=torch.tensor(h0_default),
        rho_cr_0=torch.tensor(rho_cr_0_default),
        Om0=torch.tensor(Om0_default),
    ):
        super().__init__(name)

        self.add_param("h0", h0)
        self.add_param("rho_cr_0", rho_cr_0)
        self.add_param("Om0", Om0)

        self._comoving_dist_helper_x_grid = _comoving_dist_helper_x_grid.to(
            dtype=torch.float32
        )
        self._comoving_dist_helper_y_grid = _comoving_dist_helper_y_grid.to(
            dtype=torch.float32
        )

    def dist_hubble(self, h0):
        return c_Mpc_s / (100 * km_to_mpc) / h0

    def rho_cr(self, z, x) -> torch.Tensor:
        h0, rho_cr_0, Om0 = self.unpack(x)
        Ode0 = 1 - Om0
        return rho_cr_0 * (Om0 * (1 + z) ** 3 + Ode0)

    def _comoving_dist_helper(self, x=[]):
        return interp1d(
            self._comoving_dist_helper_x_grid,
            self._comoving_dist_helper_y_grid,
            torch.atleast_1d(x),
        ).reshape(x.shape)

    def comoving_dist(self, z, x=[]):
        h0, _, Om0 = self.unpack(x)

        Ode0 = 1 - Om0
        ratio = (Om0 / Ode0) ** (1 / 3)
        return (
            self.dist_hubble(h0)
            * (
                self._comoving_dist_helper((1 + z) * ratio)
                - self._comoving_dist_helper(ratio)
            )
            / (Om0 ** (1 / 3) * Ode0 ** (1 / 6))
        )


# Lenses
class ThinLens(Parametrized):
    def __init__(self, name: str, cosmology: Cosmology, z_l: None):
        super().__init__(name)
        self.cosmology = cosmology  # this will be registered in children

        self.add_param("z_l", z_l)

    @abstractmethod
    def alpha(self, thx, thy, z_s, x=[]) -> Tuple[Tensor, Tensor]:
        ...

    def alpha_hat(self, thx, thy, z_s, x=[]) -> Tuple[Tensor, Tensor]:
        # ASSUMPTION: superclass' parameters get unpacked first
        z_l = self.unpack(x)[0]

        d_s = self.cosmology.angular_diameter_dist(z_s, x)
        d_ls = self.cosmology.angular_diameter_dist_z1z2(z_l, z_s, x)
        alpha_x, alpha_y = self.alpha(thx, thy, z_s, x)
        return (d_s / d_ls) * alpha_x, (d_s / d_ls) * alpha_y

    @abstractmethod
    def kappa(self, thx, thy, z_s, x=[]) -> Tensor:
        ...

    @abstractmethod
    def Psi(self, thx, thy, z_s, x=[]) -> Tensor:
        ...

    def Sigma(self, thx, thy, z_s, x=[]) -> Tensor:
        z_l = self.unpack(x)[0]

        Sigma_cr = self.cosmology.Sigma_cr(z_l, z_s, x)
        return self.kappa(thx, thy, z_s, x) * Sigma_cr

    def raytrace(self, thx, thy, z_s, x=[]) -> Tuple[Tensor, Tensor]:
        ax, ay = self.alpha(thx, thy, z_s, x)
        return thx - ax, thy - ay

    def time_delay(self, thx, thy, z_s, x=[]):
        z_l = self.unpack(x)[0]

        d_l = self.cosmology.angular_diameter_dist(z_l, x)
        d_s = self.cosmology.angular_diameter_dist(z_s, x)
        d_ls = self.cosmology.angular_diameter_dist_z1z2(z_l, z_s, x)
        ax, ay = self.alpha(thx, thy, z_s, x)
        Psi = self.Psi(thx, thy, z_s, x)
        factor = (1 + z_l) / c_Mpc_s * d_s * d_l / d_ls
        fp = 0.5 * d_ls**2 / d_s**2 * (ax**2 + ay**2) - Psi
        return factor * fp * arcsec_to_rad**2


class SIE(ThinLens):
    def __init__(
        self,
        name: str,
        cosmology,
        thx0=None,
        thy0=None,
        q=None,
        phi=None,
        b=None,
        z_l=None,
        s=torch.tensor(0.0),
    ):
        super().__init__(name, cosmology, z_l)

        self.add_param("thx0", thx0)
        self.add_param("thy0", thy0)
        self.add_param("q", q)
        self.add_param("phi", phi)
        self.add_param("b", b)
        self.add_param("s", s)

    def _get_psi(self, x, y, q, s):
        return (q**2 * (x**2 + s**2) + y**2).sqrt()

    def alpha(self, thx, thy, z_s, x):
        z_l, thx0, thy0, q, phi, b, s = self.unpack(x)

        s = torch.tensor(0.0, device=self.device, dtype=thx0.dtype) if s is None else s
        thx, thy = translate_rotate(thx, thy, thx0, thy0, phi)
        psi = self._get_psi(thx, thy, q, s)
        f = (1 - q**2).sqrt()
        ax = b * q.sqrt() / f * (f * thx / (psi + s)).atan()
        ay = b * q.sqrt() / f * (f * thy / (psi + q**2 * s)).atanh()

        return derotate(ax, ay, phi)

    def Psi(self, thx, thy, z_s, x):
        z_l, thx0, thy0, q, phi, b, s = self.unpack(x)

        thx, thy = translate_rotate(thx, thy, thx0, thy0, phi)
        ax, ay = self.alpha(thx, thy, z_s, x)
        return thx * ax + thy * ay

    def kappa(self, thx, thy, z_s, x):
        z_l, thx0, thy0, q, phi, b, s = self.unpack(x)

        s = torch.tensor(0.0, device=self.device, dtype=thx0.dtype) if s is None else s
        thx, thy = translate_rotate(thx, thy, thx0, thy0, phi)
        psi = self._get_psi(thx, thy, q, s)
        return 0.5 * q.sqrt() * b / psi
