# {cell=main}
using DLPipelines
using DataAugmentation
using DLPipelines: Training, Validation, Inference
using Colors: RGB
using Test
using TestSetExtensions
using StaticArrays


@testset ExtendedTestSet "`SpatialTransforms`" begin
    @testset ExtendedTestSet "image" begin
        transform = SpatialTransforms((32, 32))
        image = rand(RGB, 64, 96)

        ## We apply `SpatialTransforms` in the different [`Context`]s:
        imagetrain = transform(Training(), image)
        @test size(imagetrain) == (32, 32)

        imagevalid = transform(Validation(), image)
        @test size(imagevalid) == (32, 32)

        imageinference = transform(Inference(), image)
        @test size(imageinference) == (32, 48)

        ## During inference, the aspect ratio should stay the same
        @test size(image, 1) / size(image, 2) == size(imageinference, 1) / size(imageinference, 2)

        ## Training and validation image should be different since
        ## the first uses a random crop, and the second a center crop
        @test !all(imagetrain .≈ imagevalid)
    end

    @testset ExtendedTestSet "keypoints" begin
        transform = SpatialTransforms((32, 32))
        ks = [SVector(0, 0), SVector(64, 96)]
        keypoints = Keypoints(
            ks,
            DataAugmentation.makebounds(64, 96),
        )
        kstrain = transform(Training(), keypoints)
        ksvalid = transform(Validation(), keypoints)
        ksinference = transform(Inference(), keypoints)
        @test ksvalid[1][1] == 0
        @test ksvalid[2][1] == 32
        @test ksinference[2] == ks[2] ./ 2
    end

    @testset ExtendedTestSet "image and keypoints" begin
        transform = SpatialTransforms((32, 32))
        image = rand(RGB, 64, 96)
        ks = [SVector(0, 0), SVector(64, 96)]

        @test_nowarn transform(Training(), (image, ks))
        @test_nowarn transform(Validation(), (image, ks))
        @test_nowarn transform(Inference(), (image, ks))
    end

end

@testset ExtendedTestSet "`ImageClassification`" begin
    @testset ExtendedTestSet "Core interface" begin
        @testset ExtendedTestSet "`encodeinput`" begin
            method = ImageClassification(10, sz = (32, 32))
            image = rand(RGB, 64, 96)

            xtrain = encodeinput(method, Training(), image)
            @test size(xtrain) == (32, 32, 3)
            @test eltype(xtrain) == Float32

            xinference = encodeinput(method, Inference(), image)
            @test size(xinference) == (32, 48, 3)
            @test eltype(xinference) == Float32
        end

        @testset ExtendedTestSet "`encodetarget`" begin
            method = ImageClassification(10, sz = (32, 32))
            category = 1
            y = encodetarget(method, Training(), category)
        end

        @testset ExtendedTestSet "`encode`" begin
            method = ImageClassification(10, sz = (32, 32))
            image = rand(RGB, 64, 96)
            category = 1
            @test_nowarn encode(method, Training(), (image, category))
        end
    end
end
