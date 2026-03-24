import Pkg
using NNkernels
using ParallelTestRunner

# ENV["NNKERN_TEST_AMDGPU"] = true
# ENV["NNKERN_TEST_CUDA"] = true

if get(ENV, "NNKERN_TEST_AMDGPU", "false") == "true"
    Pkg.add("AMDGPU")
    using AMDGPU
elseif get(ENV, "NNKERN_TEST_CUDA", "false") == "true"
    Pkg.add("CUDA")
    using CUDA
else
    error("No GPU backend is set.")
end

runtests(NNkernels, ARGS)
