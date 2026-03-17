if get(ENV, "NNKERN_TEST_AMDGPU", "false") == "true"
    using AMDGPU
    kab = ROCBackend()
elseif get(ENV, "NNKERN_TEST_CUDA", "false") == "true"
    using CUDA
    kab = CUDABackend()
else
    error("No GPU backend is set.")
end
