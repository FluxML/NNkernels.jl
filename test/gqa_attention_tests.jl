using Test
using NNkernels

import Adapt
import Zygote

include("setup/core.jl")
include("setup/attention.jl")

@testset "Flash Grouped-Query Attention" begin
@testset "Grouped-Query Attention: QH=$QH, KVH=$KVH, causal=$causal, T=$T, E=$E, L=$L" for QH in (
    4, 6, 8,
), KVH in (
    1, 2,
), causal in (
    false, true,
), T in (
    Float32, Float16,
), E in (
    32, 64,
), L in (
    255, 256, 257, 512,
)
    B = 2
    q = Adapt.adapt(kab, randn(T, E, L, QH, B))
    k = Adapt.adapt(kab, randn(T, E, L, KVH, B))
    v = Adapt.adapt(kab, randn(T, E, L, KVH, B))

    o1, ∇1 = Zygote.withgradient(q, k, v) do q, k, v
        sum(naive_attention(q, k, v; causal, kpad_mask=nothing))
    end
    o2, ∇2 = Zygote.withgradient(q, k, v) do q, k, v
        sum(NNkernels.flash_attention(q, k, v; causal, kpad_mask=nothing))
    end
    eps = sizeof(T) == 4 ? 1e-3 : 2e-1
    @test isapprox(o1, o2; atol=eps, rtol=eps)
    @test isapprox(∇1[1], ∇2[1]; atol=eps, rtol=eps)
    @test isapprox(∇1[2], ∇2[2]; atol=eps, rtol=eps)
    @test isapprox(∇1[3], ∇2[3]; atol=eps, rtol=eps)
end
end
