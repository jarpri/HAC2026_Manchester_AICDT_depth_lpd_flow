"""Forward models: a shape in, the two measured curves out.

    convex_egi          Analytic operator for a convex body, linear in the facet areas, with
                        an exact transpose. No shadows, no interreflection. Used by the
                        convex stage, which gives the flow its starting support.

    mesh/               The exact physical chain on a triangle mesh, and its derivative with
                        respect to the vertices: direct light with cast shadows and penumbra
                        from a rasterised sun view (exact), interreflection by radiosity on
                        patches (radiosity), perspective rasterisation from every camera and
                        the reduction of a frame to the two numbers (raster), and the sensor
                        chain (sensor). The fitted scene and sensor parameters are the
                        Instrument (instrument). This is the model the data are compared
                        against, in calibration, training and reconstruction alike.

    shared/             Not models: the exact derivative of a hard image threshold by the
                        coarea formula (coarea), the nvdiffrast loader (_nvdr), and a slow
                        pure-torch rasteriser for tests without a GPU (software_raster).

Every model carries the lab directions into the body frame as hac26.conventions.to_body does.
"""
