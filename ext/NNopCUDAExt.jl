module NNopCUDAExt

using CUDA
using NNop

function NNop._shared_memory(::CUDABackend, device_id::Integer)
    dev = collect(CUDA.devices())[device_id]
    return UInt64(CUDA.attribute(dev, CUDA.CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK))
end

NNop.supports_wmma(::CUDABackend) = true

Base.@propagate_inbounds function NNop.wmma!(
    c::CuDeviceMatrix{T},
    a::CuDeviceMatrix{T},
    b::CuDeviceMatrix{T},
    cfg::Type{NNop.WMMATileConfig{BM, BK, BN, WM, WN, WK, aT, bT, cT}},
    tidx,
    n_warps,
    fn,
    ::Val{MAC},
) where {T, BM, BK, BN, WM, WN, WK, aT, bT, cT, MAC}
    conf = WMMA.Config{WM, WN, WK, T}
    a_ptr = pointer(a)
    b_ptr = pointer(b)
    c_ptr = pointer(c)

    a_layout = aT ? WMMA.RowMajor : WMMA.ColMajor
    b_layout = bT ? WMMA.RowMajor : WMMA.ColMajor
    c_layout = cT ? WMMA.RowMajor : WMMA.ColMajor
    a_stride = aT ? BK : BM
    b_stride = bT ? BN : BK
    c_stride = cT ? BN : BM

    n_tiles_m = BM ÷ WM
    n_tiles_n = BN ÷ WN
    n_output_tiles = n_tiles_m * n_tiles_n
    widx = (tidx - 1) ÷ 32

    # Each warp loops over multiple output tiles.
    tile_idx = widx
    while tile_idx < n_output_tiles
        tile_m = tile_idx ÷ n_tiles_n
        tile_n = tile_idx % n_tiles_n
        tile_row = tile_m * WM
        tile_col = tile_n * WN

        c_offset = cT ?
            (tile_row * BN + tile_col) :
            (tile_col * BM + tile_row)
        c_frag = MAC ?
            WMMA.load_c(c_ptr + c_offset * sizeof(T), c_stride, c_layout, conf) :
            WMMA.fill_c(zero(T), conf)

        k = 0
        while k < BK
            a_offset = aT ?
                (tile_row * BK + k) :
                (k * BM + tile_row)
            a_frag = WMMA.load_a(a_ptr + a_offset * sizeof(T), a_stride, a_layout, conf)

            b_offset = bT ?
                (k * BN + tile_col) :
                (tile_col * BK + k)
            b_frag = WMMA.load_b(b_ptr + b_offset * sizeof(T), b_stride, b_layout, conf)

            c_frag = WMMA.mma(a_frag, b_frag, c_frag, conf)
            k += WK
        end
        WMMA.store_d(c_ptr + c_offset * sizeof(T), fn(c_frag), c_stride, c_layout, conf)

        tile_idx += n_warps
    end
    return
end

end
