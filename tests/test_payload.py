"""Tests for watermark.payload — payload construction, RS coding, AES encryption."""

import numpy as np
import pytest

from watermark.payload import (
    aes_decrypt,
    aes_encrypt,
    apply_repetition_coding,
    bits_to_bytes,
    build_payload,
    bytes_to_bits,
    decode_payload,
    decode_payload_bits,
    decode_repetition_coding,
    derive_seed,
    encode_payload,
    rs_decode,
    rs_encode,
    verify_artist_id,
)


class TestBitsBytes:
    def test_roundtrip(self):
        data = b"hello world!!"
        bits = bytes_to_bits(data)
        assert len(bits) == len(data) * 8
        assert all(b in (0, 1) for b in bits)
        recovered = bits_to_bytes(bits)
        assert recovered == data

    def test_single_byte(self):
        bits = bytes_to_bits(b"\xff")
        assert np.array_equal(bits, np.ones(8, dtype=np.uint8))

    def test_zero_byte(self):
        bits = bytes_to_bits(b"\x00")
        assert np.array_equal(bits, np.zeros(8, dtype=np.uint8))


class TestReedSolomon:
    def test_no_errors(self):
        data = b"test payload data!!"
        encoded = rs_encode(data, nsym=32)
        decoded = rs_decode(encoded, nsym=32)
        assert decoded == data

    def test_corrects_errors(self):
        data = b"test payload data!!"
        encoded = bytearray(rs_encode(data, nsym=32))
        # Corrupt some bytes
        for i in range(0, 10):
            encoded[i] ^= 0xFF
        decoded = rs_decode(bytes(encoded), nsym=32)
        assert decoded == data

    def test_too_many_errors_raises(self):
        data = b"test"
        encoded = bytearray(rs_encode(data, nsym=4))
        # Corrupt more than nsym/2 symbols
        for i in range(len(encoded)):
            encoded[i] ^= 0xFF
        with pytest.raises(Exception):
            rs_decode(bytes(encoded), nsym=4)

    def test_high_redundancy_128(self):
        """Verify RS nsym=128 can correct 64 byte errors."""
        data = b"24_byte_payload_data!!!!"
        assert len(data) == 24
        encoded = bytearray(rs_encode(data, nsym=128))
        # Corrupt 60 bytes (within 64 correction limit)
        for i in range(0, 60):
            encoded[i] ^= 0xFF
        decoded = rs_decode(bytes(encoded), nsym=128)
        assert decoded == data


class TestAES:
    def test_roundtrip(self, secret_key):
        plaintext = b"secret provenance data here!!!"
        ct = aes_encrypt(plaintext, secret_key)
        assert ct != plaintext
        recovered = aes_decrypt(ct, secret_key)
        assert recovered == plaintext

    def test_no_nonce_prefix(self, secret_key):
        """AES encrypt should NOT prepend a nonce — output length equals input."""
        plaintext = b"test data here!!"
        ct = aes_encrypt(plaintext, secret_key)
        assert len(ct) == len(plaintext)

    def test_short_key_hashed(self):
        plaintext = b"test data"
        ct = aes_encrypt(plaintext, b"short")
        recovered = aes_decrypt(ct, b"short")
        assert recovered == plaintext

    def test_wrong_key_produces_garbage(self, secret_key):
        plaintext = b"secret provenance data here!!!"
        ct = aes_encrypt(plaintext, secret_key)
        # CTR mode always decrypts, but wrong key produces wrong plaintext
        wrong_result = aes_decrypt(ct, b"wrong_key_entirely_different_!!")
        assert wrong_result != plaintext

    def test_bit_error_maps_1_to_1(self, secret_key):
        """In CTR mode, flipping one ciphertext bit flips exactly one plaintext bit."""
        plaintext = b"deterministic test data!!!!!!!!"
        ct = bytearray(aes_encrypt(plaintext, secret_key))
        # Flip one bit
        ct[5] ^= 0x01
        recovered = aes_decrypt(bytes(ct), secret_key)
        # Exactly one byte should differ
        diffs = sum(a != b for a, b in zip(plaintext, recovered))
        assert diffs == 1


class TestRepetitionCoding:
    def test_no_repetition(self):
        bits = np.array([0, 1, 1, 0, 1], dtype=np.uint8)
        coded = apply_repetition_coding(bits, repetitions=1)
        assert np.array_equal(coded, bits)
        decoded = decode_repetition_coding(coded, repetitions=1)
        assert np.array_equal(decoded, bits)

    def test_triple_repetition_clean(self):
        bits = np.array([0, 1, 1, 0], dtype=np.uint8)
        coded = apply_repetition_coding(bits, repetitions=3)
        assert len(coded) == 12
        assert np.array_equal(coded, [0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0])
        decoded = decode_repetition_coding(coded, repetitions=3)
        assert np.array_equal(decoded, bits)

    def test_triple_repetition_with_errors(self):
        bits = np.array([0, 1, 1, 0], dtype=np.uint8)
        coded = apply_repetition_coding(bits, repetitions=3)
        # Corrupt one copy of each bit — majority vote should still recover
        corrupted = coded.copy()
        corrupted[0] = 1   # 0,0,0 → 1,0,0 → vote: 0
        corrupted[4] = 0   # 1,1,1 → 1,0,1 → vote: 1
        corrupted[8] = 0   # 1,1,1 → 1,1,0 → vote: 1
        corrupted[11] = 1  # 0,0,0 → 0,0,1 → vote: 0
        decoded = decode_repetition_coding(corrupted, repetitions=3)
        assert np.array_equal(decoded, bits)

    def test_five_repetition(self):
        rng = np.random.default_rng(42)
        bits = rng.integers(0, 2, size=50, dtype=np.uint8)
        coded = apply_repetition_coding(bits, repetitions=5)
        assert len(coded) == 250
        decoded = decode_repetition_coding(coded, repetitions=5)
        assert np.array_equal(decoded, bits)


class TestPayload:
    def test_build_and_decode(self, sample_rgb_image):
        raw = build_payload("alice", sample_rgb_image, timestamp=1709337600)
        assert len(raw) == 24
        prov = decode_payload(raw)
        assert prov.timestamp == 1709337600
        assert verify_artist_id("alice", prov)
        assert not verify_artist_id("bob", prov)

    def test_invalid_length_raises(self):
        with pytest.raises(ValueError):
            decode_payload(b"short")


class TestDeriveSeed:
    def test_deterministic(self):
        s1 = derive_seed(b"key1")
        s2 = derive_seed(b"key1")
        assert s1 == s2

    def test_different_keys_different_seeds(self):
        s1 = derive_seed(b"key1")
        s2 = derive_seed(b"key2")
        assert s1 != s2


class TestFullPipeline:
    def test_encode_decode_roundtrip(self, sample_rgb_image, secret_key):
        bits = encode_payload(
            "artist@email.com", sample_rgb_image, secret_key,
            timestamp=1709337600,
        )
        assert bits.dtype == np.uint8
        assert all(b in (0, 1) for b in bits)

        prov = decode_payload_bits(bits, secret_key)
        assert prov.timestamp == 1709337600
        assert verify_artist_id("artist@email.com", prov)

    def test_encode_decode_with_repetitions(self, sample_rgb_image, secret_key):
        bits = encode_payload(
            "artist@email.com", sample_rgb_image, secret_key,
            timestamp=1709337600, repetitions=3,
        )
        # With R=3, bit count should be 3x the non-repeated version
        bits_no_rep = encode_payload(
            "artist@email.com", sample_rgb_image, secret_key,
            timestamp=1709337600, repetitions=1,
        )
        assert len(bits) == 3 * len(bits_no_rep)

        prov = decode_payload_bits(bits, secret_key, repetitions=3)
        assert prov.timestamp == 1709337600
        assert verify_artist_id("artist@email.com", prov)
