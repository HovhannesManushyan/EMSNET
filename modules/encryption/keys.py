import tenseal as ts


def create_ctx(bits_scale=40, poly_mod_degree=16384, num_mul=5):
    """Helper for creating the CKKS context.
    CKKS params:
        - Polynomial degree: 8192.
        - Coefficient modulus size: [40, 21, 21, 21, 21, 21, 21, 40]. # 24 binary digit
        - Scale: 2 ** 21. # 24 ov kara
        - The setup requires the Galois keys for evaluating the convolutions.
    """

    coeff_mod_bit_sizes = [60]
    for i in range(num_mul):
        coeff_mod_bit_sizes.append(bits_scale)
    coeff_mod_bit_sizes.append(60)

    ctx = ts.context(ts.SCHEME_TYPE.CKKS, poly_mod_degree, -1, coeff_mod_bit_sizes)
    ctx.global_scale = pow(2, bits_scale)
    ctx.generate_galois_keys()

    # We prepare the context for the server, by making it public(we drop the secret key)
    server_context = ctx.copy()
    server_context.make_context_public()

    return ctx, server_context
