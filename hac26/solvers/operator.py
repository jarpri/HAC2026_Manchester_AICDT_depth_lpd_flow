"""The operator the flow is trained and run with: a raw code in, the mean-normalised curves
of every geometry out, and the gradient of a weighted residual back onto the code.

    code (dh, a), reshaping c, base support h --ImplicitBody--> field f
        --extract_mesh--> mesh
        --xy scaled by the radius--> --ExactForward--> unnormalised curves
        --normalise--> curves

The code describes the body in the canonical frame (z in [-1, 1], xy radius 1); the
measured body has the published xy radius, and its curves are not those of the canonical
body, because stretching a body sideways changes what every camera sees. So the mesh is
scaled to the physical radius before it is rendered, and the derivative runs back through
that scaling.

The gradient runs back through the same chain: the exact forward model's vector-Jacobian
product with respect to the vertices, FlexiCubes' derivative of the vertices with respect to
the field values on its grid, and the field's derivative with respect to dh and g. Nothing
in it is learned or approximated beyond the discretisation of the chain itself.

A body whose mesh is degenerate or whose radiosity patches cannot be built has no curves;
the methods return None for it and the caller decides what to do with the body.
"""
from __future__ import annotations

import torch

from hac26.conventions import PSI0
from hac26.field import (CODE_DIM, EXTRACT_EXTENT, EXTRACT_RES, N_DIR, ImplicitBody,
                         extract_mesh)
from hac26.forward.mesh.exact import ExactForward, RenderConfig, normalise, normalise_vjp
from hac26.forward.mesh.instrument import Instrument
from hac26.forward.mesh.radiosity import RadiosityError

__all__ = ["CodeOperator", "MIN_FACES"]

MIN_FACES = 8      # an extracted mesh with fewer faces is not a body


def mesh_area(verts: torch.Tensor, faces: torch.Tensor) -> float:
    """Surface area of a triangle mesh, which for a closed one is the total variation of its
    indicator: what a corrugation costs and a smooth dent does not."""
    a, b, c = (verts[faces[:, i]] for i in range(3))
    return float(0.5 * torch.cross(b - a, c - a, dim=1).norm(dim=1).sum())


def mesh_volume(verts: torch.Tensor, faces: torch.Tensor) -> float:
    """Signed volume of a closed triangle mesh by the divergence theorem."""
    a, b, c = (verts[faces[:, i]] for i in range(3))
    return float(torch.einsum("ij,ij->i", a, torch.cross(b, c, dim=1)).sum() / 6.0)


def split_code(code: torch.Tensor):
    """(dh, a) from a raw code (CODE_DIM,): the band-limited correction to the support and the
    depths on the nodes.

    The reshaping coefficients c are not part of the code. They are the degree-two part of the
    same depth field the second block carries the rest of, and a network whose blocks are two
    fixed direction sets has nowhere to put nine numbers that belong to both; a solver that fits
    c passes it beside the code."""
    if code.numel() != CODE_DIM:
        raise ValueError(f"code has {code.numel()} entries, expected CODE_DIM={CODE_DIM}")
    return code[:N_DIR], code[N_DIR:]


class CodeOperator:
    """A(x) and its adjoint for one phase grid, one instrument and one extraction resolution.

    `res` is the FlexiCubes grid the surface is extracted on. It has to resolve the angular
    scale of the depth field, which at field.N_NODES nodes is a wavelength of about a tenth of
    the body, or the body the operator renders is coarser than the one its coefficients describe
    and a fit is scored on something it is not changing. The base support `support` of every call is the origin the code's dh block
    corrects: the fitted hull of a corpus body in training, the convex stage's answer at
    reconstruction.
    """

    def __init__(self, instrument: Instrument, psi, res: int = EXTRACT_RES,
                 config: RenderConfig = RenderConfig(), device: str = "cuda",
                 backend: str | None = None):
        self.forward = ExactForward(instrument, psi, config, device=device, backend=backend)
        self.res = int(res)
        self.device = device
        self.body = ImplicitBody().to(device)

    def mesh(self, support: torch.Tensor, code: torch.Tensor, res: int | None = None,
             grad: bool = False, c: torch.Tensor | None = None):
        """The surface of the body the code describes on top of `support`, as (verts, faces)
        torch tensors on the operator's device, or None when it is degenerate. With
        `grad=True` the vertices are differentiable in `code`. `c` is the radial reshaping
        term (field.radial_field) and is absent unless it is passed."""
        self.body.set_support(support.to(self.device))
        dh, a = split_code(code.to(self.device))
        if c is not None:
            c = c.to(self.device)
        verts, faces = extract_mesh(lambda y: self.body(y, dh=dh, a=a, c=c), EXTRACT_EXTENT,
                                    res=res or self.res, device=self.device, grad=grad)
        if len(faces) < MIN_FACES:
            return None
        if not grad:
            verts = torch.as_tensor(verts, dtype=torch.float32, device=self.device)
            faces = torch.as_tensor(faces, dtype=torch.long, device=self.device)
        return verts, faces

    @staticmethod
    def canonical(verts: torch.Tensor, faces: torch.Tensor | None = None) -> torch.Tensor:
        """The challenge pose, applied to any closed mesh and differentiable in its vertices:
        z touching -1 and +1, and the largest distance from the rotation axis one. These hold
        for every body the corpus was fitted from and for every real model, so they are
        imposed on every iterate the operator sees, not only on the final answer.

        The rotation axis is the z axis of the frame the iterate is already in -- the grid the
        field is extracted on, which is centred on the origin, itself inherited from the
        convex start. It is not recentred here. The challenge fixes the axis, not the body's
        centre of mass, and the published radius is the largest distance from *the axis*; the
        released STLs are posed that way and moving them onto their own centroid moves them
        off it (hac26.shapes.rescale_touch_z). A body that is already centred, which every
        corpus body is, is unaffected.
        """
        xy = verts[:, :2]
        z = verts[:, 2]
        z = 2.0 * (z - z.min()) / (z.max() - z.min()).clamp_min(1e-9) - 1.0
        xy = xy / xy.norm(dim=1).max().clamp_min(1e-9)
        return torch.cat([xy, z[:, None]], 1)

    @classmethod
    def physical(cls, verts: torch.Tensor, faces: torch.Tensor, radius: float) -> torch.Tensor:
        """Vertices to the physical frame: posed canonically, then xy scaled by the published
        radius."""
        scale = torch.tensor([radius, radius, 1.0], dtype=verts.dtype, device=verts.device)
        return cls.canonical(verts, faces) * scale

    def curves(self, support: torch.Tensor, code: torch.Tensor, radius: float, geoms=None,
               c: torch.Tensor | None = None, res: int | None = None):
        """Mean-normalised curves (G, 2, P) of the body at the physical radius, without
        gradient, or None."""
        m = self.mesh(support, code, res=res, c=c)
        if m is None:
            return None
        try:
            return normalise(self.forward.raw_curves(self.physical(m[0], m[1], radius), m[1],
                                                     geoms=geoms, psi0=PSI0))
        except RadiosityError:
            return None

    def curves_with_shape(self, support: torch.Tensor, code: torch.Tensor, radius: float,
                          geoms=None, c: torch.Tensor | None = None, res: int | None = None):
        """`curves`, together with the surface area and the volume of the canonically posed
        body, or None.

        The two shape numbers are taken from the mesh the extraction has already built, so
        they cost a per cent of a render rather than one of their own, and they are taken in
        the canonical pose rather than the physical one because a penalty in units of area
        has to mean the same thing on a body whose published radius is 0.67 as on one whose
        radius is 3.95.
        """
        m = self.mesh(support, code, res=res, c=c)
        if m is None:
            return None
        v, f = m
        try:
            cur = normalise(self.forward.raw_curves(self.physical(v, f, radius), f,
                                                    geoms=geoms, psi0=PSI0))
        except RadiosityError:
            return None
        return cur, mesh_area(self.canonical(v, f), f), mesh_volume(self.canonical(v, f), f)

    def curves_turned(self, support: torch.Tensor, code: torch.Tensor, radius: float,
                      geoms=None):
        """The curves of `curves` together with the count curves (3, G, P) the body has under
        the thresholds of frames P/4, P/2 and 3P/4 instead of the first, in the body's own
        frame order, all mean-normalised (ExactForward.raw_curves_turned). They are what a
        quarter turn of the body needs (train_lpd.quarter_turns). None when the body has no
        curves."""
        m = self.mesh(support, code)
        if m is None:
            return None
        try:
            raw, extra = self.forward.raw_curves_turned(self.physical(m[0], m[1], radius),
                                                        m[1], geoms=geoms, psi0=PSI0)
        except RadiosityError:
            return None
        return normalise(raw), normalise(extra)

    def adjoint(self, support: torch.Tensor, code: torch.Tensor, radius: float, cot_fn,
                geoms=None):
        """The curves and the gradient, with respect to the code, of the scalar whose
        derivative with respect to the normalised curves is `cot_fn(curves)`. Returns
        (curves (G, 2, P), grad_code (CODE_DIM,)), or (None, None) when the body has no
        curves. One forward pass without gradient gives the curves; a second, chunked, pass
        carries the cotangent back to the vertices, and autograd takes it on to the code."""
        with torch.enable_grad():          # the caller may be inside no_grad
            code = code.detach().to(self.device).requires_grad_(True)
            m = self.mesh(support, code, grad=True)
            if m is None:
                return None, None
            verts, faces = m
            phys = self.physical(verts, faces, radius)
            try:
                raw, grad_v, _ = self.forward.vjp(
                    phys, faces, lambda r: normalise_vjp(r, cot_fn(normalise(r))),
                    geoms=geoms, psi0=PSI0)
            except RadiosityError:
                return None, None
            grad_code = torch.autograd.grad(phys, code, grad_outputs=grad_v)[0]
        return normalise(raw), grad_code.detach()
