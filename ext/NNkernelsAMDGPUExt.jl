module NNkernelsAMDGPUExt

using AMDGPU
using AMDGPU.Device: WMMA, ROCDeviceMatrix
using NNkernels

function NNkernels._shared_memory(::ROCBackend, device_id::Integer)
    dev = AMDGPU.devices()[device_id]
    return UInt64(AMDGPU.HIP.properties(dev).sharedMemPerBlock)
end

# TODO support rdna4, only rdna3 for now.
function NNkernels.WMMA.is_available(::ROCBackend)
    _arch_str = first(split(AMDGPU.HIP.gcn_arch(AMDGPU.device()), ':'))
    gfx = parse(Int, _arch_str[4:end])
    is_rdna3 = 1100 ≤ gfx < 1200
    return is_rdna3
end

NNkernels.WMMA.conf(::Type{<: ROCDeviceMatrix}, WM, WN, WK, T) = return nothing
NNkernels.WMMA.load_a(::Type{<: ROCDeviceMatrix}, ptr, stride, ::Val{TR}, conf) where TR =
    WMMA.load_a(ptr, stride, TR ? WMMA.RowMajor : WMMA.ColMajor)
NNkernels.WMMA.load_b(::Type{<: ROCDeviceMatrix}, ptr, stride, ::Val{TR}, conf) where TR =
    WMMA.load_b(ptr, stride, TR ? WMMA.RowMajor : WMMA.ColMajor)
NNkernels.WMMA.load_c(::Type{<: ROCDeviceMatrix}, ptr, stride, ::Val{TR}, conf) where TR =
    WMMA.load_c(ptr, stride, TR ? WMMA.RowMajor : WMMA.ColMajor)
NNkernels.WMMA.store_d(::Type{<: ROCDeviceMatrix}, ptr, frag, stride, ::Val{TR}, conf) where TR =
    WMMA.store_d(ptr, frag, stride, TR ? WMMA.RowMajor : WMMA.ColMajor)
NNkernels.WMMA.fill_c(::Type{<: ROCDeviceMatrix}, ::Type{T}, val, conf) where T = WMMA.fill_c(T, val)
NNkernels.WMMA.apply(::Type{<: ROCDeviceMatrix}, a_frag, b_frag, c_frag, conf) = WMMA.mma(a_frag, b_frag, c_frag)

end
