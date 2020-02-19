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
      [1.0],
      [1.0],
      [-1.0],
      [-1.0],
    ]

    input_to_classify = [
      [1.3, 3.0],
      [5.5, 5.5],
      [0.0, 2.0],
      [7.7, 6.0]
    ]

    model = Neuratron::DeepModel.new(2, 3)
    model.train_classification(inputs, expected_outputs, 1, 0.1)

    predictions = input_to_classify.map do |input|
      puts "Predict for #{input}"
      results = model.predict(input, Neuratron::DeepModel::Regression.new) # TODO : use Neuratron::DeepModel::Classification
      puts "Prediction #{results}"
      results
    end

    positions = input_to_classify.zip(predictions).map do |data|
      { data[0][0], data[0][1], data[1][0] }
    end

    pp positions

    # math_formulat = "#{model.weights[0]} * 1 + #{model.weights[1]} * x + #{model.weights[2]} * y"
    fns = [
      AquaPlot::Scatter3D.from_points(positions).tap(&.set_title("Points")),
      AquaPlot::Function.new("0", title: "0"),
      # AquaPlot::Function.new(math_formulat, title: math_formulat),
    ]

    pp fns[-1].style = "pm3d"

    plt = AquaPlot::Plot3D.new fns
    plt.set_key("left box")
    plt.show
    plt.set_view(100, 80, 1)
    plt.show
    plt.close
  end
end
