CONTEXTS = (Training(), Validation(), Inference())


"""
    checkmethod(method, sample, model)

Check if `method` conforms to the `DLPipelines.jl` interfaces.
`sample` and `model` are used for testing.
"""
function checkmethod(method, sample, model; device = identity)
    model = device(model)
    @testset "DLPipelines.jl interfaces" begin

        @testset "Core interface" begin
            @testset "`encode`" begin
                for context in CONTEXTS
                    @test_nowarn encode(method, context, sample)
                end
            end

            @testset "`encode!`" begin
                for context in CONTEXTS
                    buf = encode(method, context, sample)
                    @test_nowarn encode!(buf, method, context, sample)
                end
            end

            @testset "Model compatibility" begin
                for context in CONTEXTS
                    x, y = encode(method, context, sample)
                    @test_nowarn y_hat = _predictx(method, model, x, device)
                end
            end

            @testset "`decodeŷ" begin
                for context in CONTEXTS
                    x, y = encode(method, context, sample)
                    y_hat = _predictx(method, model, x, device)
                    @test_nowarn decodeŷ(method, context, y_hat)
                end
            end
        end

        @testset "Interpretation interface" begin
            context = Training()
            x, y = encode(method, context, sample)
            ŷ = _predictx(method, model, x, device)
            target = decodeŷ(method, context, ŷ)
            @test_nowarn interpretsample(method, sample)
            @test_nowarn interpretx(method, x)
            @test_nowarn interprety(method, y)
            @test_nowarn interpretŷ(method, ŷ)
            @test_nowarn interpretstep(method, x, y, ŷ)
        end
    end
end


"""
    checkmethod_core(method; model = mockmodel(method), sample = mocksample(method), devicefn)

Checks if `method` implements the core interface.
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
