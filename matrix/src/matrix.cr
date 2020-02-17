require "aquaplot"

@[Link("neuratron")]
lib LibNeuratron
  @[Extern]
  struct LinearModel
    inputs: Float64*
    size_input: Int32
    size_output: Int32
  end

  fun create_linear_model(input : Int32, size: Int32) : Pointer(LinearModel)
  fun train_linear_model(
    model : LinearModel*,
    input : Float64*, input_size : Int32,
    output : Float64*, output_size : Int32
  ) : Bool

  fun predict_linear_model(model : LinearModel*, input : Float64*) : Float64
end

module Neuratron
  class LinearModel
    @model : LibNeuratron::LinearModel*

    def initialize(@input_size : Int32, @ouput_size : Int32)
      @model = LibNeuratron.create_linear_model(input_size, ouput_size)
    end

    def model
      @model.value
    end

    def weights
      (0...((model.size_input + 1) * model.size_output)).map do |i|
        model.inputs[i]
      end
    end

    def train(input, output)
      LibNeuratron.train_linear_model(@model, input.to_unsafe, input.size, output.to_unsafe, output.size)
    end

    def predict(input)
      LibNeuratron.predict_linear_model(@model, input.to_unsafe)
    end
  end
end

inputs = [
  [1.0, 2.0],
  [3.0, 4.0],
  [5.0, 5.0],
  [6.0, 6.0],
]
expected_outputs = [
  [2.0],
  [3.0],
  [1.0],
  [40.0],
]

model = Neuratron::LinearModel.new(2, 1)
model.train(inputs.flatten, expected_outputs.flatten)
pp model.weights

predictions = inputs.map do |input|
  puts "Predict for #{input}"
  results = model.predict(input)
  puts "Prediction #{results}"
  results
end

pp predictions

positions = inputs.zip(predictions).map do |data|
  { data[1], data[0][1], data[0][0] }
end

pp "positions", positions


math_formulat = "#{model.weights[0]}x + #{model.weights[1]}y + #{model.weights[2]}"
puts "math_formulat #{math_formulat}"
fns = [
  AquaPlot::Function.new("#{model.weights[0]} * x + #{model.weights[1]} * y + #{model.weights[2]}", title: "regression"),
  AquaPlot::Scatter3D.from_points(positions)
]

pp fns[0].style = "pm3d"

plt = AquaPlot::Plot3D.new fns
plt.set_key("left box")
plt.show
plt.close
