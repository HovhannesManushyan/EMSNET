import tenseal as ts


# Helper for encoding the image
def encrypt_vector(ctx, vector):
    enc_input = ts.ckks_vector(ctx, vector)
    return enc_input


def deserialize_enc_vector(context: bytes, ckks_vector: bytes) -> ts.CKKSVector:
    try:
        ctx = ts.context_from(context)
        enc_x = ts.ckks_vector_from(ctx, ckks_vector)
    except:
        raise ValueError("cannot deserialize context or ckks_vector")
    try:
        _ = ctx.galois_keys()
    except:
        raise ValueError("the context doesn't hold galois keys")
    return enc_x
