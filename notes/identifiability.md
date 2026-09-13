# What the curves choose, and what chooses the hull

`notes/representation.md` measures what the correction can hold and `notes/photometry.md`
what the rig measures. This note records two questions that were put to the data directly, one
answered and one answered against the hypothesis that prompted it.

## The body is a minimum of the misfit, and the wrong body is another one

Body 3 was written in the fit's own coordinates -- the nine reshaping coefficients that carry
the convex answer's hull onto the body's own hull, solved on points sampled from that hull,
and the depths that carve the rest -- and the fit was started there. Curves are
through the stand-in renderer described below, at 24 phases and 96 pixels over the 21 distinct
geometries and both curve types; `chi` is their root mean square difference from the released
Blender curves and `dice` the voxel overlap with the released shape at 128 cubed.

| body | chi | dice |
|---|---|---|
| the convex stage's answer | 0.0782 | 0.7075 |
| the fit from that answer | 0.0376 | 0.7205 |
| the body itself, in these coordinates | 0.0349 | 0.9888 |
| the fit released from the body | 0.0307 | 0.9584 |

Released from the body the fit stays within 0.03 of the overlap it started with and lowers the
misfit, and the misfit it reaches is below what the fit reaches from the convex answer. The
body's neighbourhood is a basin of the misfit and it is the deeper of the two, so the
objective is the right one and the gap between what the fit returns and the body is a gap in
the search.

The two basins are separated by 0.0069, and the renderer used to measure them misses the
released curves at the released mesh by 0.011. The ordering is real but finer than the
instrument that measured it, which is why the residual `scripts/calibrate.py` reports is the
first number to read, and why `scripts/select_answers.py` decides every body on cameras its
own fit never saw rather than on the misfit it was fitted to.

## The hull cannot be had from the unshadowed geometries

At zero phase angle the camera looks along the illumination, so every visible point is lit and
nothing shadows anything. The total Lambert flux is then the projected area of the silhouette,
and the silhouette of a body is the silhouette of its convex hull, so such a curve measures
the hull and says nothing about the concavities. The rig has one geometry at zero phase and
two at 21 degrees, and the residual of the convex stage's answer against the released curves
of body 3 grows monotonically with phase angle, from 0.011 at zero to 0.149 at 135 degrees,
which is the shadowing entering curves a convex body has nothing to explain it with.

That suggests estimating the hull from the low-phase geometries alone, where the estimate
cannot be biased by concavity. Measured, it is worse. A convex inversion of body 3 restricted
to phase angles below 25 degrees returns a body 1.46 times the volume of its own hull and
overlapping it by 0.757; admitting every geometry returns 1.23 and 0.831. Three geometries
sweep three circles of projection directions as the body turns, and three circles do not
determine a convex body: what the restriction removes in bias it more than loses in coverage,
and the smoothness prior fills the rest of the sphere in.

The overlap available from the hull alone is worth recording, since it bounds what any
correction of the hull can buy: body 3's own convex hull overlaps it by 0.869, against the
convex stage's 0.708. Slightly more than half of what the convex answer loses is hull and the
rest is the concavity.

## The stand-in

These numbers are through an orthographic ray cast with a Lambert surface, cast shadows and
the measured power-law transfer, which reproduces the released Blender curves at the released
meshes to about 0.011 root mean square. It stands in for the chain in `hac26/forward/mesh`,
which needs a GPU. The chain's own residual at the released shapes is what the calibration
reports and is not measured here.
