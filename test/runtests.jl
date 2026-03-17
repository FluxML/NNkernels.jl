import Pkg
using NNkernels
using Test
using ReTestItems

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

nworkers = clamp(Sys.CPU_THREADS ÷ 2, 1, 4)
runtests(NNkernels; nworkers, nworker_threads=1, testitem_timeout=60 * 15) do ti
    return true
end
