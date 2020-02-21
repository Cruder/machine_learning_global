require "./spec_helper"

describe Neuratron::LinearModel do
  it "with 4 exemples" do
    inputs = [
      [1.0, 2.0],
      [3.0, 4.0],
      [5.0, 5.0],
      [6.0, 6.0],
    ]
    expected_outputs = [
      [-1.0, 1.0, -1.0],
      [-1.0, 1.0, -1.0],
      [1.0, -1.0, -1.0],
      [-1.0, -1.0, 1.0],
    ]

    input_to_classify = [
    #   [1.3, 3.0],
    #   [5.5, 5.5],
    #   [0.0, 2.0],
    #   [7.7, 6.0]
    # ]
      [1.0, 2.0],
      [3.0, 4.0],
      [5.0, 5.0],
      [6.0, 6.0],
    ]
    model = Neuratron::DeepModel.new(2, 3, 3, 3)
    model.train_classification(inputs, expected_outputs, 500, 0.01)
    predictions = input_to_classify.map do |input|
      puts "Predict for #{input}"
      results = model.predict(input)
      puts "Prediction #{results}"
      results
    end
    points = Array(Array(Tuple(Float64, Float64))).new(3) { Array(Tuple(Float64, Float64)).new }
    positions = input_to_classify.zip(predictions).map do |data|
      pp data[1]
      points[max_indice(data[1])] << { data[0][0], data[0][1] }
    end
    aqua_points = Array(AquaPlot::Scatter).new
    points.each do |point|
      aqua_points << AquaPlot::Scatter.from_points(point).tap(&.set_title("Points"))
    end
    #pp aqua_points[-1].style = "pm3d"

    plt = AquaPlot::Plot.new aqua_points
    plt.set_key("off box")
    plt.show
    # plt.set_view(100, 80, 1)
    # plt.show
    plt.close

    model.save("./save/tmp.json")
  end
end
