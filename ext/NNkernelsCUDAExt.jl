module NNkernelsCUDAExt

using CUDA
using NNkernels

function NNkernels._shared_memory(::CUDABackend, device_id::Integer)
    dev = collect(CUDA.devices())[device_id]
    return UInt64(CUDA.attribute(dev, CUDA.CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK))
end

NNkernels.WMMA.is_available(::CUDABackend) = true

NNkernels.WMMA.conf(
    ::Type{<: CuDeviceMatrix}, ::Val{WM}, ::Val{WN}, ::Val{WK}, ::Type{T},
) where {WM, WN, WK, T}= WMMA.Config{WM, WN, WK, T}
NNkernels.WMMA.load_a(::Type{<: CuDeviceMatrix}, ptr, stride, ::Val{TR}, conf) where TR =
    WMMA.load_a(ptr, stride, TR ? WMMA.RowMajor : WMMA.ColMajor, conf)
NNkernels.WMMA.load_b(::Type{<: CuDeviceMatrix}, ptr, stride, ::Val{TR}, conf) where TR =
    WMMA.load_b(ptr, stride, TR ? WMMA.RowMajor : WMMA.ColMajor, conf)
NNkernels.WMMA.load_c(::Type{<: CuDeviceMatrix}, ptr, stride, ::Val{TR}, conf) where TR =
    WMMA.load_c(ptr, stride, TR ? WMMA.RowMajor : WMMA.ColMajor, conf)
NNkernels.WMMA.store_d(::Type{<: CuDeviceMatrix}, ptr, frag, stride, ::Val{TR}, conf) where TR =
    WMMA.store_d(ptr, frag, stride, TR ? WMMA.RowMajor : WMMA.ColMajor, conf)
NNkernels.WMMA.fill_c(::Type{<: CuDeviceMatrix}, ::Type{T}, val, conf) where T = WMMA.fill_c(val, conf)
NNkernels.WMMA.apply(::Type{<: CuDeviceMatrix}, a_frag, b_frag, c_frag, conf) = WMMA.mma(a_frag, b_frag, c_frag, conf)

end
