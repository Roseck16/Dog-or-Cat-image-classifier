using DelimitedFiles

folder_test = "dataset/test_set"

test_data = training_data(folder_test, 2000, 1, shape)
test_data = cpu.(test_data)
model = cpu(model)

test_acc = accuracy(test_data[1][1], test_data[1][2])
# 0.7825 13 27
# 0.791 15 3
# 0.787 20 4

using Plots

data = readdlm("losses.txt")
y = data[:,1]
x = round.(Int64, data[:,2])

p = plot(x,y, title="Loss training for 27 epochs", label="Loss")

savefig(p, "plot_loss.png")

data_acc = readdlm("accuracy.txt")

y = data_acc[:,1]
x = data_acc[:,2]

p = plot(x,y, title="Accuracy training for 27 epochs", label="")
savefig(p, "plot_accuracy.png")
