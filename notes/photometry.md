# What the rig measures

The organisers reduce every video frame to two numbers, and everything in this repository
that renders a body has to produce those two numbers rather than something adjacent to them.
This note derives what they are, measures the one constant they contain, and records where
the code had them wrong.

## The two reductions

A pixel records the radiance of the surface its ray meets, through the camera's transfer. For
a surface of uniform albedo lit by a collimated beam, Lambert's law gives that radiance as
proportional to the illumination cosine and independent of the direction the surface is seen
from,

    L = (rho E / pi) V mu0,      mu0 = max(0, n . s),

with V the binary shadow visibility. Write T for the monotone transfer from radiance to
stored value. The intensity curve sums the stored values above a low fixed level; the count
curve counts the pixels above Otsu's level of that video's own first frame, held fixed for
the rotation.

Two consequences follow and they are the whole of what makes this tractable.

Because T is monotone and L is proportional to mu0, the count is the projected measure of a
super-level set of the illumination cosine over the lit, visible, unshadowed surface. No
property of the transfer, the exposure, the quantisation or the albedo survives into it
except through one level. The count curve is geometric; it is not a photometric quantity.

If T is a power law of exponent gamma, the intensity curve is the gamma-th moment of the
illumination cosine over the visible lit projected area. Only at gamma = 1 is that the
disk-integrated brightness the asteroid literature inverts. Measured on the released render
gamma is about a half, so the curve weighs the terminator and the grazing limb far more
heavily than a brightness would, and a model that computes a brightness and then applies a
power law to the *curve* rather than to the *image* is computing something else.

## The kernels

For a facet of area A, unit normal n, seen with mu = n.v and lit with mu0 = n.s, the radiance
does not depend on v, the facet covers A mu of the image, and each of its pixels carries the
stored value mu0**gamma. So

    intensity      A mu mu0**gamma
    count          A mu  where mu0 > c

and the emission cosine enters only as the area a facet covers. This is checkable without any
data: on a convex body nothing shadows, so the facet sum is the whole forward model and a ray
cast of the same body must agree with it exactly.

On the convex hull of body 3, over the 21 distinct geometries at 24 phases and 192 pixels,
the two agree to a median of 0.0011 on the intensity curves and 0.0038 on the counts. The
kernel this repository used before -- Lommel-Seeliger plus a Lambert term, with the count
thresholded at zero on the product of the two cosines -- disagrees with the same cast by
0.0504 and 0.0786, and by 0.10 and 0.18 at the geometries with the longest shadows. Those
last are of the same order as the entire difference between body 3 and its own convex hull,
so a convex inversion under the old kernel had to distort the shape to reproduce the curves.
No Lambert weight in the grid the conventions once searched repairs it, because that family
does not contain mu mu0**gamma.

## The level the count is thresholded at

Otsu's criterion needs only the histogram of a frame. For a body written as facets that
histogram is known: a lit visible facet contributes its projected area at the stored value
mu0**gamma, and the rest of the frame contributes at zero. So the level is derived rather
than fitted, which matters because a level free to move is a level that can absorb a shape
error.

Derived this way it reproduces the level a rendered first frame of the same body gives to
0.002 of the illumination cosine at every one of the 21 geometries, and it ranges from 0.16
at zero phase angle to 0.05 at the largest, so a single constant would be wrong. It is
insensitive to how much of the frame the body is set in: over framings from one and a half to
five times the body's own silhouette the level moves by under 0.003.

## The camera table

The published table gives the top camera's elevation at each azimuth. Rendering the released
body 3 at several elevations and comparing with its released curves recovers the published
value at six of the seven azimuths, and at azimuth 135 prefers 24 degrees where the table says
26, by a factor of nearly four in the residual of that column:

| azimuth | published | 21 deg | 24 deg | 26 deg | 28 deg |
|---|---|---|---|---|---|
| 0 | 21 | **0.0008** | 0.0044 | 0.0078 | 0.0112 |
| 45 | 26 | 0.0088 | 0.0039 | **0.0010** | 0.0039 |
| 90 | 26 | 0.0177 | 0.0074 | **0.0019** | 0.0076 |
| 135 | 26 | 0.0353 | **0.0066** | 0.0249 | 0.0469 |
| 225 | 24 | 0.0224 | **0.0066** | 0.0177 | 0.0313 |
| 270 | 24 | 0.0070 | **0.0021** | 0.0047 | 0.0084 |
| 315 | 24 | 0.0038 | **0.0010** | 0.0027 | 0.0052 |

The residual is parabolic about the winner at every azimuth, so the measurement resolves the
elevation to well under a degree. Azimuth 135 is one of the two geometries at the largest
phase angle, where the shadows are longest.

## What follows for the instrument

For the rendered channel there is nothing left to fit. The camera is at infinity, the source
is a parallel beam, there is no bounce light, no point spread, no lens falloff and no spline
transfer, because a render has none of those; the one constant is the exponent, and it belongs
to the channel rather than to any body. Everything a calibration would otherwise be free to
move is a direction a fit would use to absorb an error of shape: a penumbra is a blur of the
shadow edge, which is the feature that carries concavity, and a free exponent flattens a
curve's peaks exactly as a body too large in one direction does.

The laboratory channel is a different instrument and none of this applies to it. It has a
lens, a sensor and bounce light off a matte print, and it is fitted.
