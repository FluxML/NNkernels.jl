module NNopMetalExt

using Metal
using NNop

function NNop._shared_memory(::MetalBackend, device_id::Integer)
    dev = Metal.devices()[device_id]
    return UInt64(dev.maxThreadgroupMemoryLength)
end

end
