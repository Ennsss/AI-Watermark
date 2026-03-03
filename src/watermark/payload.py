"""F2: Provenance payload builder — artist ID, timestamp, pHash, RS coding, AES-256."""

from __future__ import annotations

import hashlib
import struct
import time
from typing import NamedTuple

import numpy as np
from Crypto.Cipher import AES
from PIL import Image
from reedsolo import RSCodec

# Default Reed-Solomon symbol count — corrects up to nsym/2 symbol errors.
# With nsym=128 on a 24-byte message → 152 bytes total, corrects 64 symbol errors (42%).
DEFAULT_RS_NSYM = 128


class ProvenanceData(NamedTuple):
    """Decoded provenance payload."""

    artist_id: str
    timestamp: int
    phash: int


def derive_seed(key: bytes) -> int:
    """Derive a deterministic PRNG seed from a secret key.

    Uses SHA-256 of the key, takes first 8 bytes as uint64.

    Args:
        key: Secret key bytes (any length).

    Returns:
        Integer seed suitable for numpy PRNG.
    """
    digest = hashlib.sha256(key).digest()
    return struct.unpack(">Q", digest[:8])[0]


def compute_phash(image: np.ndarray) -> int:
    """Compute perceptual hash of an RGB image.

    Args:
        image: (H, W, 3) uint8 RGB array.

    Returns:
        64-bit integer perceptual hash.
    """
    import imagehash

    pil_img = Image.fromarray(image)
    h = imagehash.phash(pil_img, hash_size=8)
    return int(str(h), 16)


def build_payload(artist_id: str, image: np.ndarray, timestamp: int | None = None) -> bytes:
    """Construct a binary provenance payload.

    Layout (24 bytes total):
      - artist_id_hash: 8 bytes (SHA-256 of artist_id, truncated)
      - timestamp:      8 bytes (uint64, UTC epoch seconds)
      - phash:          8 bytes (uint64, perceptual hash)

    Args:
        artist_id: Artist identifier string (name, email, UUID).
        image: (H, W, 3) uint8 RGB image for pHash computation.
        timestamp: UTC epoch seconds. If None, uses current time.

    Returns:
        24-byte raw payload.
    """
    if timestamp is None:
        timestamp = int(time.time())

    # Hash artist ID to fixed 8 bytes
    id_hash = hashlib.sha256(artist_id.encode("utf-8")).digest()[:8]

    # Perceptual hash
    phash = compute_phash(image)

    # Pack as big-endian: 8B id_hash + 8B timestamp + 8B phash
    payload = id_hash + struct.pack(">Q", timestamp) + struct.pack(">Q", phash)
    return payload


def decode_payload(raw: bytes) -> ProvenanceData:
    """Decode a raw 24-byte payload into provenance fields.

    Note: artist_id is returned as the hex of the truncated hash,
    since the original string is not recoverable. Comparison is done
    by hashing the candidate artist ID and checking against this value.

    Args:
        raw: 24-byte payload.

    Returns:
        ProvenanceData with (artist_id_hex, timestamp, phash).
    """
    if len(raw) != 24:
        raise ValueError(f"Expected 24-byte payload, got {len(raw)}")
    id_hash = raw[:8].hex()
    timestamp = struct.unpack(">Q", raw[8:16])[0]
    phash = struct.unpack(">Q", raw[16:24])[0]
    return ProvenanceData(artist_id=id_hash, timestamp=timestamp, phash=phash)


def verify_artist_id(candidate_id: str, payload: ProvenanceData) -> bool:
    """Check if a candidate artist ID matches the payload hash.

    Args:
        candidate_id: Artist ID string to verify.
        payload: Decoded provenance data.

    Returns:
        True if the candidate ID's hash matches the payload.
    """
    id_hash = hashlib.sha256(candidate_id.encode("utf-8")).digest()[:8].hex()
    return id_hash == payload.artist_id


def rs_encode(data: bytes, nsym: int = DEFAULT_RS_NSYM) -> bytes:
    """Apply Reed-Solomon error correction encoding.

    Args:
        data: Raw payload bytes.
        nsym: Number of RS symbols (higher = more redundancy).

    Returns:
        RS-encoded bytes (data + parity).
    """
    codec = RSCodec(nsym)
    encoded = codec.encode(data)
    return bytes(encoded)


def rs_decode(data: bytes, nsym: int = DEFAULT_RS_NSYM) -> bytes:
    """Apply Reed-Solomon error correction decoding.

    Args:
        data: RS-encoded bytes (possibly corrupted).
        nsym: Must match encoding nsym.

    Returns:
        Corrected original payload bytes.

    Raises:
        reedsolo.ReedSolomonError: If too many errors to correct.
    """
    codec = RSCodec(nsym)
    decoded = codec.decode(data)
    return bytes(decoded[0])


def aes_encrypt(plaintext: bytes, key: bytes) -> bytes:
    """Encrypt data with AES-256-CTR.

    Uses CTR mode so that bit errors in ciphertext cause exactly one bit
    error in plaintext (no propagation). Critical for watermarking where
    we expect some bit corruption after compression.

    The nonce is deterministically derived from the key (not embedded in
    the bitstream), so corrupted bits cannot destroy the nonce.

    Args:
        plaintext: Data to encrypt.
        key: 32-byte AES key. If shorter, SHA-256 hashed to 32 bytes.

    Returns:
        Ciphertext (same length as plaintext). No nonce prefix.
    """
    if len(key) != 32:
        key = hashlib.sha256(key).digest()
    # Deterministic nonce derived from key — recomputed on decrypt side
    nonce = hashlib.sha256(key + b"nonce").digest()[:8]
    cipher = AES.new(key, AES.MODE_CTR, nonce=nonce)
    return cipher.encrypt(plaintext)


def aes_decrypt(ciphertext: bytes, key: bytes) -> bytes:
    """Decrypt AES-256-CTR data.

    CTR mode: bit errors map 1:1 from ciphertext to plaintext,
    so Reed-Solomon can correct them effectively.

    Nonce is recomputed from the key (not read from ciphertext),
    avoiding catastrophic failure when nonce bits are corrupted.

    Args:
        ciphertext: Encrypted data (no nonce prefix).
        key: 32-byte AES key. If shorter, SHA-256 hashed to 32 bytes.

    Returns:
        Decrypted plaintext.
    """
    if len(key) != 32:
        key = hashlib.sha256(key).digest()
    nonce = hashlib.sha256(key + b"nonce").digest()[:8]
    cipher = AES.new(key, AES.MODE_CTR, nonce=nonce)
    return cipher.decrypt(ciphertext)


def apply_repetition_coding(bits: np.ndarray, repetitions: int = 1) -> np.ndarray:
    """Repeat each bit R times for redundancy against high BER.

    Args:
        bits: 1D uint8 array of 0s and 1s.
        repetitions: Number of times to repeat each bit. 1 = no repetition.

    Returns:
        1D uint8 array of length len(bits) * repetitions.
    """
    if repetitions <= 1:
        return bits
    return np.repeat(bits, repetitions)


def decode_repetition_coding(bits: np.ndarray, repetitions: int = 1) -> np.ndarray:
    """Decode repetition-coded bits via majority vote.

    Args:
        bits: 1D uint8 array with R consecutive copies of each original bit.
        repetitions: Number of repetitions used during encoding.

    Returns:
        1D uint8 array of decoded bits (length = len(bits) / repetitions).
    """
    if repetitions <= 1:
        return bits
    n = len(bits) // repetitions
    reshaped = bits[: n * repetitions].reshape(n, repetitions)
    # Majority vote: >50% ones → 1, else 0
    return (reshaped.sum(axis=1) > repetitions / 2).astype(np.uint8)


def encode_payload(
    artist_id: str,
    image: np.ndarray,
    key: bytes,
    timestamp: int | None = None,
    rs_nsym: int = DEFAULT_RS_NSYM,
    repetitions: int = 1,
) -> np.ndarray:
    """Full payload encoding pipeline: build → RS encode → AES encrypt → to bits → repeat.

    Args:
        artist_id: Artist identifier string.
        image: (H, W, 3) uint8 RGB image for pHash.
        key: Secret key for AES encryption and PRNG seeding.
        timestamp: Optional UTC epoch. If None, uses current time.
        rs_nsym: Reed-Solomon redundancy symbols.
        repetitions: Repetition coding factor (1 = none, 3 = each bit repeated 3x).

    Returns:
        1D uint8 array of bits (0s and 1s) ready for embedding.
    """
    raw = build_payload(artist_id, image, timestamp)
    rs_encoded = rs_encode(raw, nsym=rs_nsym)
    encrypted = aes_encrypt(rs_encoded, key)
    bits = bytes_to_bits(encrypted)
    return apply_repetition_coding(bits, repetitions)


def decode_payload_bits(
    bits: np.ndarray,
    key: bytes,
    rs_nsym: int = DEFAULT_RS_NSYM,
    repetitions: int = 1,
) -> ProvenanceData:
    """Full payload decoding pipeline: majority vote → bytes → AES decrypt → RS decode → parse.

    Args:
        bits: 1D uint8 array of extracted bits.
        key: Secret key for AES decryption.
        rs_nsym: Reed-Solomon redundancy symbols (must match encoding).
        repetitions: Repetition coding factor (must match encoding).

    Returns:
        ProvenanceData with recovered provenance fields.

    Raises:
        ValueError: If decryption or RS decoding fails.
    """
    bits = decode_repetition_coding(bits, repetitions)
    encrypted = bits_to_bytes(bits)
    rs_encoded = aes_decrypt(encrypted, key)
    raw = rs_decode(rs_encoded, nsym=rs_nsym)
    return decode_payload(raw)


def bytes_to_bits(data: bytes) -> np.ndarray:
    """Convert bytes to a bit array.

    Args:
        data: Input bytes.

    Returns:
        1D uint8 array of 0s and 1s, length = len(data) * 8.
    """
    arr = np.frombuffer(data, dtype=np.uint8)
    return np.unpackbits(arr)


def bits_to_bytes(bits: np.ndarray) -> bytes:
    """Convert a bit array back to bytes.

    Args:
        bits: 1D uint8 array of 0s and 1s. Length must be multiple of 8.

    Returns:
        Bytes object.
    """
    if len(bits) % 8 != 0:
        # Pad to multiple of 8
        pad_len = 8 - (len(bits) % 8)
        bits = np.concatenate([bits, np.zeros(pad_len, dtype=np.uint8)])
    return np.packbits(bits).tobytes()
