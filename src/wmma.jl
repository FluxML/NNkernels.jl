module WMMA

struct TileConfig{BM, BK, BN, WM, WN, WK, aT, bT, cT} end

is_available(kab) = false

function conf end
function load_a end
function load_b end
function load_c end
function fill_c end
function apply end
function store_d end

Base.@propagate_inbounds function wmma!(
    c::AbstractMatrix{T},
    a::AbstractMatrix{T},
    b::AbstractMatrix{T},
    cfg::Type{TileConfig{BM, BK, BN, WM, WN, WK, aT, bT, cT}},
    tidx,
    n_warps,
    fn,
    ::Val{MAC},
) where {T, BM, BK, BN, WM, WN, WK, aT, bT, cT, MAC}
    AT = typeof(a)
    wmma_cfg = conf(AT, Val(WM), Val(WN), Val(WK), T)

    a_ptr = pointer(a)
    b_ptr = pointer(b)
    c_ptr = pointer(c)

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
            load_c(AT, c_ptr + c_offset * sizeof(T), c_stride, Val(cT), wmma_cfg) :
            fill_c(AT, Float32, zero(Float32), wmma_cfg)

        k = 0
        while k < BK
            a_offset = aT ?
                (tile_row * BK + k) :
                (k * BM + tile_row)
            a_frag = load_a(AT, a_ptr + a_offset * sizeof(T), a_stride, Val(aT), wmma_cfg)

            b_offset = bT ?
                (k * BN + tile_col) :
                (tile_col * BK + k)
            b_frag = load_b(AT, b_ptr + b_offset * sizeof(T), b_stride, Val(bT), wmma_cfg)

            c_frag = apply(AT, a_frag, b_frag, c_frag, wmma_cfg)
            k += WK
        end
        store_d(AT, c_ptr + c_offset * sizeof(T), fn(c_frag), c_stride, Val(cT), wmma_cfg)
        tile_idx += n_warps
    end
    return
end

end
