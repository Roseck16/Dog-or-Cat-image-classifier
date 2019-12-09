using Images
using Flux, CuArrays
using Base.Iterators: partition

function natural(x, y) # Function  to sort the files so that "ab2" < "ab10"
    k(x) = [occursin(r"\d+", s) ? parse(Int, s) : s
            for s in split(replace(x, r"\d+" => s->" $s "))]
    A = k(x); B= k(y)
    for (a, b) in zip(A, B)
        if !isequal(a, b)
            return typeof(a) <: typeof(b) ? isless(a, b) :
                   isa(a,Int) ? true : false
        end
    end
    return length(A) < length(B)
end

getarray(x, reshape_size) = Float64.(permutedims(channelview(imresize(Images.load(x), reshape_size)), (2,3,1)))

function get_names(folder::String, batch_size::Int64, start::Int64)
    names = []
    for subfolder in readdir(folder)
        i = 0
        files = readdir("$folder/$subfolder")
        files = sort!(files, lt=natural)
        for file in start:length(files)
            if i < batch_size/2
                i += 1
                push!(names, folder*"/"*subfolder*"/"*files[file])
            else
                break
            end
        end
    end
    return names
end

function load_images(names::Array{Any, 1}, reshape_size::Tuple = (256, 256))
    imgs = [getarray(names[i], reshape_size) for i in 1:size(names,1)]
end

function load_labels(names::Array{Any, 1})
    labels = []
    for item in names
        if occursin("cats", item)
            push!(labels, 0,1)
        else
            push!(labels, 1,0)
        end
    end
    return reshape(labels, 2,:)
end

function training_data(folder, batch_size, start, shape)
    names_images = get_names(folder, batch_size, start)
    imgs = load_images(names_images, shape)
    labels = load_labels(names_images)
    train = gpu.([(cat(imgs[i]..., dims=4), labels[:,i]) for i in partition(1:size(names_images, 1), batch_size)])
    train = gpu(train)
end
