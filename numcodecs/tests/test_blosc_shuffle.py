from multiprocessing import Pool
from multiprocessing.pool import ThreadPool


import numpy as np
import pytest


try:
    from numcodecs import blosc
    from numcodecs.blosc import Blosc, Shuffle
except ImportError:  # pragma: no cover
    pytest.skip(
        "numcodecs.blosc not available", allow_module_level=True
    )


from numcodecs.tests.common import (check_encode_decode,
                                    check_config,
                                    check_backwards_compatibility,
                                    check_err_decode_object_buffer,
                                    check_err_encode_object_buffer,
                                    check_max_buffer_size)


codecs = [
    Shuffle(shuffle=Blosc.SHUFFLE),
    Shuffle(shuffle=Blosc.SHUFFLE, blocksize=0),
    Shuffle(shuffle=Blosc.SHUFFLE, blocksize=2**8),
]


# mix of dtypes: integer, float, bool, string
# mix of shapes: 1D, 2D, 3D
# mix of orders: C, F
arrays = [
    np.arange(1000, dtype='i4'),
    np.linspace(1000, 1001, 1000, dtype='f8'),
    np.random.normal(loc=1000, scale=1, size=(100, 10)),
    np.random.randint(0, 2, size=1000, dtype=bool).reshape(100, 10, order='F'),
    np.random.choice([b'a', b'bb', b'ccc'], size=1000).reshape(10, 10, 10),
    np.random.randint(0, 2**60, size=1000, dtype='u8').view('M8[ns]'),
    np.random.randint(0, 2**60, size=1000, dtype='u8').view('m8[ns]'),
    np.random.randint(0, 2**25, size=1000, dtype='u8').view('M8[m]'),
    np.random.randint(0, 2**25, size=1000, dtype='u8').view('m8[m]'),
    np.random.randint(-2**63, -2**63 + 20, size=1000, dtype='i8').view('M8[ns]'),
    np.random.randint(-2**63, -2**63 + 20, size=1000, dtype='i8').view('m8[ns]'),
    np.random.randint(-2**63, -2**63 + 20, size=1000, dtype='i8').view('M8[m]'),
    np.random.randint(-2**63, -2**63 + 20, size=1000, dtype='i8').view('m8[m]'),
]


@pytest.fixture(scope='module', params=[True, False, None])
def use_threads(request):
    return request.param


@pytest.mark.parametrize('array', arrays)
@pytest.mark.parametrize('codec', codecs)
def test_encode_decode(array, codec):
    check_encode_decode(array, codec)


def test_config():
    codec = Shuffle(shuffle=Blosc.SHUFFLE)
    check_config(codec)
    codec = Shuffle(shuffle=Blosc.SHUFFLE, blocksize=2**8)
    check_config(codec)


def test_repr():
    expect = "Shuffle(shuffle=SHUFFLE, blocksize=0)"
    actual = repr(Shuffle(shuffle=Blosc.SHUFFLE, blocksize=0))
    assert expect == actual
    expect = "Shuffle(shuffle=SHUFFLE, blocksize=256)"
    actual = repr(Shuffle(shuffle=Blosc.NOSHUFFLE, blocksize=256))
    assert expect == actual
    expect = "Shuffle(shuffle=BITSHUFFLE, blocksize=512)"
    actual = repr(Shuffle(shuffle=Blosc.BITSHUFFLE, blocksize=512))
    assert expect == actual
    expect = "Shuffle(shuffle=AUTOSHUFFLE, blocksize=1024)"
    actual = repr(Shuffle(shuffle=Blosc.AUTOSHUFFLE,
                        blocksize=1024))
    assert expect == actual


def test_eq():
    assert Blosc() == Blosc()
    assert Blosc(shuffle=SHUFFLE) != Blosc(shuffle=AUTOSHUFFLE)
    assert Blosc(shuffle=SHUFFLE) != Blosc(shuffle=BITSHUFFLE)


def test_shuffle_blocksize_default(use_threads):
    arr = np.arange(1000, dtype='i4')

    blosc.use_threads = use_threads

    # default blocksize
    enc = blosc.blosc_shuffle(arr, Blosc.SHUFFLE)
    _, _, blocksize = blosc.cbuffer_sizes(enc)
    assert blocksize > 0

    # explicit default blocksize
    enc = blosc.blosc_shuffle(arr, Blosc.SHUFFLE, 0)
    _, _, blocksize = blosc.cbuffer_sizes(enc)
    assert blocksize > 0


@pytest.mark.parametrize('bs', (2**7, 2**8))
def test_shuffle_blocksize(use_threads, bs):
    arr = np.arange(1000, dtype='i4')

    blosc.use_threads = use_threads

    enc = blosc.blosc_shuffle(arr, Blosc.SHUFFLE, bs)
    _, _, blocksize = blosc.cbuffer_sizes(enc)
    assert blocksize == bs


def _encode_worker(data):
    compressor = Suffle(shuffle=Blosc.SHUFFLE)
    enc = compressor.encode(data)
    return enc


def _decode_worker(enc):
    compressor = Suffle(shuffle=Blosc.SHUFFLE)
    data = compressor.decode(enc)
    return data


@pytest.mark.parametrize('pool', (Pool, ThreadPool))
def test_multiprocessing(use_threads, pool):
    data = np.arange(1000000)
    enc = _encode_worker(data)

    pool = pool(5)

    try:
        blosc.use_threads = use_threads

        # test with process pool and thread pool

        # test encoding
        enc_results = pool.map(_encode_worker, [data] * 5)
        assert all([len(enc) == len(e) for e in enc_results])

        # test decoding
        dec_results = pool.map(_decode_worker, [enc] * 5)
        assert all([data.nbytes == len(d) for d in dec_results])

        # tidy up
        pool.close()
        pool.join()

    finally:
        blosc.use_threads = None  # restore default


def test_err_decode_object_buffer():
    check_err_decode_object_buffer(Blosc())


def test_err_encode_object_buffer():
    check_err_encode_object_buffer(Blosc())


def test_decompression_error_handling():
    for codec in codecs:
        with pytest.raises(RuntimeError):
            codec.decode(bytearray())
        with pytest.raises(RuntimeError):
            codec.decode(bytearray(0))


def test_max_buffer_size():
    for codec in codecs:
        assert codec.max_buffer_size == 2**31 - 1
        check_max_buffer_size(codec)
