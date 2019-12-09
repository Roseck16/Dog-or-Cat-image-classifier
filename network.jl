using CuArrays
using Flux, DelimitedFiles
import Flux: crossentropy, binarycrossentropy
import Statistics: mean
import BSON: @save, @load

_batch_size = 64
data_size = 8000
folder = "dataset/training_set"
shape = (64, 64)

last_epoch = 27 #15 13
last_batch = 27 #3 27

include("image_preprocessing.jl")

#_train = training_data(folder, _batch_size, 1, shape)

function round_array(array)
    rounded = []
    for i in array
        push!(rounded, round(Int32, i))
    end
    return reshape(rounded, size(array))
end

if isfile("checkpoints/model_epoch-$(last_epoch)_batch-$(last_batch).bson")

    @load "checkpoints/model_epoch-$(last_epoch)_batch-$(last_batch).bson" local_model
    model = gpu(local_model)
    ps = params(model)
    @info("Model loaded from epoch $last_epoch and batch $last_batch")
else

    @info("Creating model")

    model = Chain(
    Conv((3,3), 3=>64, relu, pad=(1,1), stride=(1,1)),
    BatchNorm(64),
    MaxPool((2,2)),
    Conv((3,3), 64=>64, relu, pad=(1,1), stride=(1,1)),
    BatchNorm(64),
    MaxPool((2,2)),
    Conv((3,3), 64=>128, relu, pad=(1,1), stride=(1,1)),
    BatchNorm(128),
    MaxPool((2,2)),
    Conv((3,3), 128=>128, relu, pad=(1,1), stride=(1,1)),
    BatchNorm(128),
    MaxPool((2,2)),
    Conv((3,3), 128=>256, relu, pad=(1,1), stride=(1,1)),
    BatchNorm(256),
    MaxPool((2,2)),

    x -> reshape(x, :, size(x,4)),
    Dense(1024, 512, relu),
    Dropout(0.5),
    Dense(512, 2),
    softmax
    ) |> gpu

    ps = params(model)

    @info("Model Created")
end

function loss(x, y)
    ŷ = model(x)
    crossentropy(ŷ, y)
end

function accuracy(x, y, _model)
    ŷ = round_array(_model(x))
    mean(ŷ .== y)
end

CuArrays.allowscalar(true)

opt = ADAM()
worst_loss = 0.0003631989

function train(epochs::Int64, last_epoch::Int64, folder::String, batch_size::Int64, last_batch::Int64, shape)
    batches = round(Int64, ((data_size)/2)/batch_size)
    _last_epoch = last_epoch
    _last_batch = last_batch
    _step = 1036

    for epoch = _last_epoch:epochs
        @info("\n- - - - - - - Epoch $epoch - - - - - - -")
        batch = _last_batch
        place = (batch*batch_size) + 1

        for i in place:batch_size:round(Int64, data_size/2)
            batch += 1

            train_data = training_data(folder, batch_size, i, shape)

            Flux.train!(loss, ps, train_data, opt)
            _loss = loss(train_data[1][1], train_data[1][2])
            _acc = accuracy(train_data[1][1], train_data[1][2], model)

            open("losses.txt", "a") do lo
                writedlm(lo, [_loss _step])
            end
            open("accuracy.txt", "a") do ac
                writedlm(ac, [_acc _step])
            end
            _step += 1
            if batch % 5 == 0 || batch == 1 || batch == batches
                global model
                @info("Epoch: $(epoch) - Batch: [$(batch)/$(batches)] - Loss: $(_loss) - Accuracy: $(_acc)")
                local_model = cpu(model)
                if batch == 1 || batch == batches
                    @save "checkpoints/model_epoch-$(epoch)_batch-$(batch).bson" local_model
                end
            end
            if _loss < worst_loss
                global model
                local_model = cpu(model)
                test_acc = accuracy(test_data[1][1], test_data[1][2], local_model)
                @info("Epoch: $(epoch) - Batch: [$(batch)/$(batches)] - Loss: $(_loss) - Test accuracy: $(test_acc)")
                @info("New best loss! Saving model...")
                @save "checkpoints/model_epoch-$(epoch)_batch-$(batch).bson" local_model
                global worst_loss = _loss
            end
        end
        _last_batch = 0
        _last_epoch += 1
    end
end

train(100, last_epoch, folder, _batch_size, last_batch, shape)
