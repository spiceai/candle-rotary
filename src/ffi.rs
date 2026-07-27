use core::ffi::{c_int, c_long, c_void};

extern "C" {
    pub(crate) fn candle_rotary_embedding(
        // `query` and `key` are rotated in place by the kernel; the caches are read-only.
        query: *mut c_void,
        key: *mut c_void,
        cos_cache: *const c_void,
        sin_cache: *const c_void,

        is_neox: c_int,

        head_size: c_int,
        num_tokens: c_long,
        rot_dim: c_int,
        num_heads: c_int,
        num_kv_heads: c_int,
        query_stride: c_long,
        key_stride: c_long,

        dtype: u32,
    );
}
