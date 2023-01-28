import tenseal as ts


def create_ctx(bits_scale=26, poly_mod_degree=16384):
    """Helper for creating the CKKS context.
    CKKS params:
        - Polynomial degree: 8192.
        - Coefficient modulus size: [40, 21, 21, 21, 21, 21, 21, 40].
        - Scale: 2 ** 21.
        - The setup requires the Galois keys for evaluating the convolutions.
    """

    coeff_mod_bit_sizes = [31, bits_scale, bits_scale, bits_scale, bits_scale, bits_scale, bits_scale, bits_scale,
                           bits_scale, 31]
    ctx = ts.context(ts.SCHEME_TYPE.CKKS, poly_mod_degree, -1, coeff_mod_bit_sizes)
    ctx.global_scale = pow(2, bits_scale)
    ctx.generate_galois_keys()

    # We prepare the context for the server, by making it public(we drop the secret key)
    server_context = ctx.copy()
    server_context.make_context_public()

    return ctx, server_context
