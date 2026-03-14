module NNopAMDGPUExt

using AMDGPU
using AMDGPU.Device: WMMA
using NNop

function NNop._shared_memory(::ROCBackend, device_id::Integer)
    dev = AMDGPU.devices()[device_id]
    return UInt64(AMDGPU.HIP.properties(dev).sharedMemPerBlock)
end

# Only RDNA 3 for now.
function NNop.supports_wmma(::ROCBackend)
    _arch_str = first(split(AMDGPU.HIP.gcn_arch(AMDGPU.device()), ':'))
    gfx = parse(Int, _arch_str[4:end])
    is_rdna3 = 1100 ≤ gfx < 1200
    return is_rdna3
end

Base.@propagate_inbounds function NNop.wmma!(
    c::AMDGPU.Device.ROCDeviceMatrix{T},
    a::AMDGPU.Device.ROCDeviceMatrix{T},
    b::AMDGPU.Device.ROCDeviceMatrix{T},
    cfg::Type{NNop.WMMATileConfig{BM, BK, BN, WM, WN, WK, aT, bT, cT}},
    tidx,
    n_warps,
    fn,
    ::Val{MAC},
) where {T, BM, BK, BN, WM, WN, WK, aT, bT, cT, MAC}
    a_ptr = pointer(a)
    b_ptr = pointer(b)
    c_ptr = pointer(c)

    a_layout = aT ? WMMA.RowMajor() : WMMA.ColMajor()
    b_layout = bT ? WMMA.RowMajor() : WMMA.ColMajor()
    c_layout = cT ? WMMA.RowMajor() : WMMA.ColMajor()
    a_stride = Int32(aT ? BK : BM)
    b_stride = Int32(bT ? BN : BK)
    c_stride = Int32(cT ? BN : BM)

    n_tiles_m = BM ÷ WM
    n_tiles_n = BN ÷ WN
    n_output_tiles = n_tiles_m * n_tiles_n
    widx = (tidx - 1) ÷ 32

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
            WMMA.load_c(c_ptr + c_offset * sizeof(T), c_stride, c_layout) :
            WMMA.fill_c(Float32, zero(Float32))

        k = 0
        while k < BK
            a_offset = aT ?
                (tile_row * BK + k) :
                (k * BM + tile_row)
            a_frag = WMMA.load_a(a_ptr + a_offset * sizeof(T), a_stride, a_layout)

            b_offset = bT ?
                (k * BN + tile_col) :
                (tile_col * BK + k)
            b_frag = WMMA.load_b(b_ptr + b_offset * sizeof(T), b_stride, b_layout)

            c_frag = WMMA.mma(a_frag, b_frag, c_frag)
            k += WK
        end
        WMMA.store_d(c_ptr + c_offset * sizeof(T), fn(c_frag), c_stride, c_layout)

        tile_idx += n_warps
    end
    return
end

end
