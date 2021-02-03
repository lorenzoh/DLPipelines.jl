CONTEXTS = (Training(), Validation(), Inference())


"""
    checkmethod(method, sample, model; device = identity)
    checkmethod(method; device = identity)

Check if `method` conforms to the `DLPipelines.jl` interfaces.
`sample` and `model` are used for testing. If you have implemented the testing
interface and don't supply these as arguments, `mocksample(method)` and
`mockmodel(method)` will be used.

Checks *core* and *interpretation* interfaces.

"""
function checkmethod(method, sample, model; kwargs...)
    checkmethod_core(method, sample, model; kwargs...)
    checkmethod_interpretation(method, sample, model; kwargs...)
end


"""
    checkmethod_core(method, sample, model; device = identity)
    checkmethod_core(method; device = identity)

Check if `method` conforms to the [core interface](docs/interfaces/core.md).
`sample` and `model` are used for testing. If you have implemented the testing
interface and don't supply these as arguments, `mocksample(method)` and
`mockmodel(method)` will be used.
"""
function checkmethod_core(
        method;
        model = mockmodel(method),
        sample = mocksample(method),
        devicefn = identity)
    @testset "Core interface" begin
        @testset "`encode`" begin
            for context in CONTEXTS
                @test_nowarn encode(method, context, sample)
            end
        end
        @testset "Model compatibility" begin
            for context in CONTEXTS
                x, y = encode(method, context, sample)
                @test_nowarn ŷ = _predictx(method, model, x, devicefn)
            end
        end
        @testset "`decodeŷ" begin
            for context in CONTEXTS
                x, y = encode(method, context, sample)
                ŷ = _predictx(method, model, x, devicefn)
                @test_nowarn decodeŷ(method, context, ŷ)
            end
        end
    end
end

"""
    checkmethod_interpretation(method, sample, model; device = identity)
    checkmethod_interpretation(method; device = identity)

Check if `method` conforms to the [core interface](docs/interfaces/core.md).
`sample` and `model` are used for testing. If you have implemented the testing
interface and don't supply these as arguments, `mocksample(method)` and
`mockmodel(method)` will be used.
"""
function checkmethod_interpretation(
        method;
        model = mockmodel(method),
        sample = mocksample(method),
        devicefn = identity)

    @testset "Interpretation interface" begin
        context = Training()
        x, y = encode(method, context, sample)
        ŷ = _predictx(method, model, x, devicefn)
        target = decodeŷ(method, context, ŷ)
        @testset "`interpretsample`" begin
            @test_nowarn interpretsample(method, sample)
        end
        @testset "`interpretx`" begin
            @test_nowarn interpretx(method, x)
        end
        @testset "`interprety`" begin
            @test_nowarn interprety(method, y)
        end
        @testset "`interpretŷ`" begin
            @test_nowarn interpretŷ(method, ŷ)
        end
        @testset "`interprettarget`" begin
            @test_nowarn interprettarget(method, target)
        end
    end

end
